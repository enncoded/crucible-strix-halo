// Real dense weights: isolate stream issue, decode ALU, stores, and phase launches.
#define main DenseHarnessMain
#include "mega_dense.hip"
#undef main

namespace ablate {
constexpr int Slots = 3, Chunk = 4, MaxK = 6, SlotWords = 8 * Chunk * 8 * MaxK;
struct Phase { const LinJob* jobs; const GemvItem* items; int n; };

template <int L, int C>
using Ring = LdsStreamRing<Slots, L, C>;

template <int K, int L, int C>
__device__ __forceinline__ void Loader(const LinJob& job, const GemvItem& item,
                                      uint32_t* data, Ring<L,C>& ring, uint32_t base) {
  constexpr int T = 8*K, ColVec = Chunk*T/4, NV = (8*ColVec+32*L-1)/(32*L);
  const int lane = threadIdx.x&31, lw = threadIdx.x>>5;
  const int A = job.I/16, span = A/job.S, a0 = item.s*span;
  const int count = (span+Chunk-1)/Chunk;
  const auto rs = StreamRsrc(job.tiles, (uint32_t)(size_t)A*(job.O/16)*T*4);
  auto ld = [&](int seq, int j) -> uint4 {
    const int vi = j*32*L+lw*32+lane, col=vi/ColVec, off=vi%ColVec;
    const int vo = ((item.b0+col)*A+a0)*T*4+off*16;
    const auto raw = __builtin_amdgcn_raw_buffer_load_b128(rs, vo, seq*Chunk*T*4, 0);
    return *reinterpret_cast<const uint4*>(&raw);
  };
#define AL_ONE(R,S,J) if constexpr(NV>J) R##J=ld(S,J)
#define AL_LD(R,S) AL_ONE(R,S,0);AL_ONE(R,S,1);AL_ONE(R,S,2);AL_ONE(R,S,3);AL_ONE(R,S,4);AL_ONE(R,S,5);AL_ONE(R,S,6);AL_ONE(R,S,7);AL_ONE(R,S,8);AL_ONE(R,S,9);AL_ONE(R,S,10);AL_ONE(R,S,11)
#define AL_STORE(R,J) if constexpr(NV>J) d[J*32*L]=R##J
#define AL_PUT(R,S) if((S)<count){const uint32_t g=base+(S);ring.WaitWritable(g);uint4*d=reinterpret_cast<uint4*>(data+(g%Slots)*SlotWords)+lw*32+lane;AL_STORE(R,0);AL_STORE(R,1);AL_STORE(R,2);AL_STORE(R,3);AL_STORE(R,4);AL_STORE(R,5);AL_STORE(R,6);AL_STORE(R,7);AL_STORE(R,8);AL_STORE(R,9);AL_STORE(R,10);AL_STORE(R,11);ring.Publish(g,lane);}
  uint4 A0,A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11,B0,B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11;
  AL_LD(A,0);
  for(int seq=0;seq<count;seq+=2){AL_LD(B,seq+1);AL_PUT(A,seq);AL_LD(A,seq+2);AL_PUT(B,seq+1);}
#undef AL_PUT
#undef AL_STORE
#undef AL_LD
#undef AL_ONE
}

template <int K,int L,int C,int Mode>
__device__ __forceinline__ uint32_t Consumer(const LinJob& job,const GemvItem& item,
                                            const uint32_t*data,Ring<L,C>&ring,
                                            const uint32_t*xs,uint32_t base,uint32_t sink){
  constexpr int T=8*K, E=8/C;
  const int lane=threadIdx.x&31,cw=__builtin_amdgcn_readfirstlane((threadIdx.x>>5)-L);
  const int span=job.I/16/job.S,q=lane&3;
  const LaneWindow<K> win(lane);
  float acc[E][2]={};
  for(int seq=0,a=0;a<span;++seq,a+=Chunk){
    const uint32_t g=base+seq;
    ring.WaitReady(g);
    if constexpr(Mode==0){
      const uint4*raw=reinterpret_cast<const uint4*>(data+(g%Slots)*SlotWords);
#pragma unroll
      for(int j=0;j<(8*Chunk*T/4+32*C-1)/(32*C);++j){
        const int i=j*32*C+cw*32+lane;
        if(i<8*Chunk*T/4){const uint4 v=raw[i];sink^=v.x^v.y^v.z^v.w;}
      }
    }else{
#pragma unroll
      for(int e=0;e<E;++e){
        const int col=cw+e*C;
        if(col<item.nb){
          const uint32_t*column=data+(g%Slots)*SlotWords+col*Chunk*T;
#pragma unroll
          for(int f=0;f<Chunk;++f){
            if(a+f>=span)break;
            const uint32_t*tile=column+f*T;
            __half2 P[4];
            DecodeLane<K,kMulMode>(tile[win.d0],tile[(win.d0+1)%T],tile[(win.d0+2)%T],win.shift,P);
            const uint32_t xa=xs[(a+f)*8+q],xb=xs[(a+f)*8+4+q];
            const __half2 ha=*reinterpret_cast<const __half2*>(&xa),hb=*reinterpret_cast<const __half2*>(&xb);
            acc[e][0]=Dot2(P[0],ha,acc[e][0]);acc[e][0]=Dot2(P[1],hb,acc[e][0]);
            acc[e][1]=Dot2(P[2],ha,acc[e][1]);acc[e][1]=Dot2(P[3],hb,acc[e][1]);
          }
        }
      }
    }
    ring.Release(g,lane);
  }
  if constexpr(Mode!=0){
#pragma unroll
    for(int e=0;e<E;++e){
      if(cw+e*C<item.nb){
        float a=acc[e][0],b=acc[e][1];
        a=a+__shfl_xor(a,1);a=a+__shfl_xor(a,2);
        b=b+__shfl_xor(b,1);b=b+__shfl_xor(b,2);
        sink^=__float_as_uint(a)^__float_as_uint(b);
        if constexpr(Mode==2){if(q==0){const int col=16*(item.b0+cw+e*C)+lane/4;
          float*out=job.partial+(size_t)item.s*job.O;out[col]=a;out[col+8]=b;}}
      }
    }
  }
  return sink;
}

template <int L,int C,int Mode,bool Load,size_t N=0>
__device__ __forceinline__ uint32_t Dispatch(int K,const LinJob&job,const GemvItem&item,
                                            uint32_t*data,Ring<L,C>&ring,uint32_t*xs,
                                            uint32_t base,uint32_t sink){
  if(K==kSupportedDecodeK[N]){
    if constexpr(Load) Loader<kSupportedDecodeK[N],L,C>(job,item,data,ring,base);
    else sink=Consumer<kSupportedDecodeK[N],L,C,Mode>(job,item,data,ring,xs,base,sink);
  }else if constexpr(N+1<kSupportedDecodeK.size())sink=Dispatch<L,C,Mode,Load,N+1>(K,job,item,data,ring,xs,base,sink);
  return sink;
}

template <int C>
__device__ __forceinline__ void ComputeMeet(uint32_t*arrived,uint32_t&seq,int t){
  const uint32_t target=++seq*C;
  asm volatile("s_waitcnt lgkmcnt(0)":::"memory");
  if((t&31)==0)__hip_atomic_fetch_add(arrived,1u,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_WORKGROUP);
  while(__hip_atomic_load(arrived,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_WORKGROUP)<target)__builtin_amdgcn_s_sleep(1);
  asm volatile("":::"memory");
  __builtin_amdgcn_wave_barrier();
}

template <int L,int C,int Mode>
__global__ void __launch_bounds__(32*(L+C)) Kernel(const Phase*phases,int nph,uint32_t*sinks){
  __shared__ Ring<L,C> ring;
  __shared__ uint32_t data[Slots*SlotWords],arrived;
  extern __shared__ uint32_t xs[];
  ring.Initialize(threadIdx.x);
  if(threadIdx.x==0)arrived=0;
  __syncthreads();
  const bool load=threadIdx.x<32*L;
  const int t=threadIdx.x-32*L;
  uint32_t seq=0,meet=0,sink=0;
  if(load){
    for(int p=0;p<nph;++p){const Phase ph=StreamConstant(phases+p);
      for(int it=blockIdx.x;it<ph.n;it+=gridDim.x){
        const GemvItem item=StreamConstant(ph.items+it);const LinJob job=StreamConstant(ph.jobs+item.job);
        sink=Dispatch<L,C,Mode,true>(job.K,job,item,data,ring,xs,seq,sink);
        seq+=(job.I/16/job.S+Chunk-1)/Chunk;
      }
    }
  }else{
    for(int p=0;p<nph;++p){const Phase ph=StreamConstant(phases+p);
      for(int it=blockIdx.x;it<ph.n;it+=gridDim.x){
        const GemvItem item=StreamConstant(ph.items+it);const LinJob job=StreamConstant(ph.jobs+item.job);
        const int span=job.I/16/job.S;const uint32_t*x=reinterpret_cast<const uint32_t*>(job.xh);
        for(int i=t;i<span*8;i+=32*C)xs[i]=x[item.s*span*8+i];
        ComputeMeet<C>(&arrived,meet,t);
        sink=Dispatch<L,C,Mode,false>(job.K,job,item,data,ring,xs,seq,sink);
        seq+=(span+Chunk-1)/Chunk;
        ComputeMeet<C>(&arrived,meet,t);
      }
    }
    sinks[(size_t)blockIdx.x*32*C+t]=sink;
  }
}

using Fn=void(*)(const Phase*,int,uint32_t*);
template<int L,int C>Fn PickMode(int mode){return mode==0?Kernel<L,C,0>:mode==1?Kernel<L,C,1>:Kernel<L,C,2>;}
Fn Pick(int l,int c,int m){
  if(c==4){if(l==1)return PickMode<1,4>(m);if(l==2)return PickMode<2,4>(m);return PickMode<4,4>(m);}
  if(l==1)return PickMode<1,8>(m);if(l==2)return PickMode<2,8>(m);return PickMode<4,8>(m);
}
}

int main(int argc,char**argv){
  if(argc<2){std::fprintf(stderr,"usage: mega_ablate CKPT [grid=80]\n");return 2;}
  const int grid=argc>2?std::atoi(argv[2]):80;
  if(grid<1||grid>1024)return 2;
  st::Checkpoint ck(argv[1]);auto groups=LoadGroups(ck,true);
  std::mt19937 rng(17);std::normal_distribution<float>norm;
  std::vector<__half>hx(10240);for(auto&h:hx)h=__float2half(norm(rng));
  __half*x;float*partial;uint32_t*sinks;
  HIP_CHECK(hipMalloc(&x,hx.size()*2));HIP_CHECK(hipMemcpy(x,hx.data(),hx.size()*2,hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&partial,64<<20));HIP_CHECK(hipMalloc(&sinks,(size_t)grid*256*4));
  std::vector<Plan>plans;std::vector<ablate::Phase>ph;
  size_t bytes=0,lds=0;
  for(const auto&g:groups){plans.push_back(MakePlan(g,x,partial));const auto&p=plans.back();bytes+=p.bytes;lds=std::max(lds,p.lds);ph.push_back({p.dj,p.di,(int)p.items.size()});}
  ablate::Phase*dph;HIP_CHECK(hipMalloc(&dph,ph.size()*sizeof(ablate::Phase)));
  HIP_CHECK(hipMemcpy(dph,ph.data(),ph.size()*sizeof(ablate::Phase),hipMemcpyHostToDevice));
  std::printf("# phases=%zu bytes=%.9fGB grid=%d dynLDS=%zu roof=245.5GB/s\n",ph.size(),bytes/1e9,grid,lds);
  for (int c : {4,8}) for (int l : {1,2,4}) for (int i : {0,1,6,7,96}) {
    const Plan& p=plans[i];
    std::vector<float> reference(p.partials),actual(p.partials);
    Launch(p,false,grid);
    HIP_CHECK(hipMemcpy(reference.data(),partial,p.partials*4,hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemset(partial,0xff,p.partials*4));
    hipLaunchKernelGGL(ablate::Pick(l,c,2),grid,32*(l+c),lds,nullptr,dph+i,1,sinks);
    HIP_CHECK(hipMemcpy(actual.data(),partial,p.partials*4,hipMemcpyDeviceToHost));
    if(std::memcmp(reference.data(),actual.data(),p.partials*4)){
      std::fprintf(stderr,"full mismatch L%d C%d phase%d\n",l,c,i);return 1;
    }
  }
  std::printf("# full partials exact for all six wave splits on five representative phases\n");
  for(int c:{4,8})for(int l:{1,2,4})for(int mode:{0,1,2})for(bool persistent:{false,true}){
    const auto fn=ablate::Pick(l,c,mode);int resident;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fn,32*(l+c),lds));
    std::vector<double>times;
    for(int rep=0;rep<6;++rep){bench::EventTimer timer;timer.Begin();
      if(persistent)hipLaunchKernelGGL(fn,grid,32*(l+c),lds,nullptr,dph,(int)ph.size(),sinks);
      else for(size_t i=0;i<ph.size();++i)hipLaunchKernelGGL(fn,grid,32*(l+c),lds,nullptr,dph+i,1,sinks);
      const float ms=timer.EndMs();HIP_CHECK(hipGetLastError());if(rep)times.push_back(ms);
    }
    const double ms=bench::Median(times),bw=bytes/ms/1e6;
    std::printf("L%d C%d %-8s %-10s %8.3fms %7.2fGB/s %6.2f%%DRAM resident=%d\n",l,c,mode==0?"bytes":mode==1?"no-store":"full",persistent?"persistent":"per-phase",ms,bw,bw/245.5*100,resident*20);
    std::fflush(stdout);
  }
}
