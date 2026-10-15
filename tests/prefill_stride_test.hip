// Exercise physical padding with nonzero canaries and exact bulk arithmetic.
#define main StrideProbeMain
#include "../bench/prefill_glue_stride.hip"
#undef main
#include "../engine/layers/prefill_ple.h"
#include "../engine/exl3/prefill_linear.h"

static void Equal(const std::vector<float>& a,const std::vector<float>& b) {
  if (a.size()!=b.size() || std::memcmp(a.data(),b.data(),a.size()*sizeof(float))) {
    std::fprintf(stderr,"FAIL: padded result differs\n"); std::exit(1);
  }
}
static std::vector<float> Tight(const Buffer<float>& h,int rows,int stride) {
  const auto raw=h.Get(); std::vector<float> out((size_t)rows*kWide);
  for (int r=0;r<rows;++r) std::copy_n(raw.data()+(size_t)r*stride,kWide,out.data()+(size_t)r*kWide);
  for (int r=0;r<rows;++r) for (int d=kWide;d<stride;++d)
    if (raw[(size_t)r*stride+d]!=123.f) { std::fprintf(stderr,"FAIL: padding write\n"); std::exit(1); }
  return out;
}
static void Check(int R) {
  constexpr int pad=64,M=kRank+4,ldh=kWide+pad;
  const auto values=Random((size_t)R*kWide,.3f,29);
  std::vector<float> padded((size_t)R*ldh,123.f);
  for (int r=0;r<R;++r) std::copy_n(values.data()+(size_t)r*kWide,kWide,padded.data()+(size_t)r*ldh);
  Buffer<float> h(values),p(padded),d1((size_t)R*M*4),d2(d1.n),s1(R*4),s2(s1.n);
  Buffer<__half> fn(Half(Random((size_t)M*kWide,.02f,31)));
  auto dots=[&](float* H,float* d,float* s,int stride) {
    PrefillHcDotsWsKernel<64,0,true,2,2,false,4><<<4*((R+63)/64),288>>>(fn.p,M,H,d,s,R,stride);
  };
  dots(h.p,d1.p,s1.p,kWide); dots(p.p,d2.p,s2.p,ldh); CheckLaunch();
  Equal(d1.Get(),d2.Get()); Equal(s1.Get(),s2.Get());
  Buffer<__half> up(Half(Random((size_t)kWide*kRank,.02f,37)));
  Buffer<__half> thi(Half(Random((size_t)R*kRank,.02f,41))),tlo(Half(Random((size_t)R*kRank,.002f,43)));
  Buffer<float> norm(std::vector<float>(kWide,1.f)),m1((size_t)R*kHidden),m2(m1.n);
  for (int pass=0;pass<2;++pass)
    PrefillHcMixWideKernel<true><<<dim3(kHidden/16,(R+127)/128),256>>>(up.p,norm.p,thi.p,tlo.p,
        pass?s2.p:s1.p,pass?p.p:h.p,pass?m2.p:m1.p,R,nullptr,pass?ldh:kWide);
  CheckLaunch(); Equal(m1.Get(),m2.Get());
  Buffer<float> pd(Random((size_t)R*kSlots*kHidden,.03f,47)),post(Random(R*4,.2f,53));
  Buffer<__half> svh(std::vector<__half>(kHidden,__float2half(1.f)));
  Buffer<int> selected(std::vector<int>(R*kTopK,0)),offsets(std::vector<int>(kExperts+1,0));
  std::vector<int> inverse(R*kTopK); for (int i=0;i<R*kTopK;++i) inverse[i]=i;
  Buffer<int> inv(inverse); std::vector<LinJob> jobs(kExperts+1);
  jobs[0].partial=pd.p; jobs[0].svh=svh.p; jobs[0].R=R*kTopK; jobs[0].O=kHidden; jobs[0].S=1;
  jobs[kExperts]=jobs[0]; jobs[kExperts].partial=pd.p+(size_t)R*kTopK*kHidden; jobs[kExperts].R=R;
  Buffer<LinJob> down(jobs); Buffer<float> weights(std::vector<float>(R*kSlots,.09f));
  PrefillMoe a{}; a.dispatch.rows=R; a.dispatch.selected=selected.p; a.dispatch.offsets=offsets.p;
  a.inverse=inv.p; a.down=down.p;
  PrefillMoeCombineKernel<false><<<(R*20+7)/8,256>>>(a,weights.p,post.p,h.p);
  PrefillMoeCombineKernel<false><<<(R*20+7)/8,256>>>(a,weights.p,post.p,p.p,ldh);
  CheckLaunch(); Equal(h.Get(),Tight(p,R,ldh));
  Buffer<__half> hpd(Half(pd.Get()));
  jobs[0].partial=reinterpret_cast<float*>(hpd.p);
  jobs[kExperts].partial=reinterpret_cast<float*>(hpd.p+(size_t)R*kTopK*kHidden);
  Buffer<LinJob> hdown(jobs); a.down=hdown.p;
  PrefillMoeCombineKernel<true><<<(R*20+7)/8,256>>>(a,weights.p,post.p,h.p);
  PrefillMoeCombineKernel<true><<<(R*20+7)/8,256>>>(a,weights.p,post.p,p.p,ldh);
  CheckLaunch(); Equal(h.Get(),Tight(p,R,ldh));
  std::printf("PASS: stride rows %d\n",R);
}
static void CheckStorage(int R) {
  constexpr int ldh=kWide+64;
  Buffer<float> h((size_t)R*kWide),p(std::vector<float>((size_t)R*ldh,123.f));
  std::vector<uint16_t> table(32*kHidden); for (size_t i=0;i<table.size();++i) table[i]=0x3f00+i%255;
  std::vector<int> tokens(R); for (int r=0;r<R;++r) tokens[r]=r%32;
  Buffer<uint16_t> embed(table); Buffer<int> tok(tokens);
  EmbedKernel<<<dim3(kHidden/256,R),256>>>(embed.p,tok.p,h.p);
  EmbedKernel<<<dim3(kHidden/256,R),256>>>(embed.p,tok.p,p.p,ldh);
  CheckLaunch(); Equal(h.Get(),Tight(p,R,ldh));
  Buffer<float> kv(Random((size_t)R*12800,.3f,59)),norm(std::vector<float>(kWide,1.f));
  Buffer<float> conv(Random(kWide*4,.1f,61)),ring1(16*kWide),ring2(16*kWide),g1(R*4),g2(R*4);
  Buffer<__half> nv1((size_t)R*kWide),nv2(nv1.n);
  PleArgs a{kv.p,norm.p,norm.p,norm.p,conv.p,ring1.p,h.p,0,R};
  auto b=a; b.ring=ring2.p; b.H=p.p; b.ldh=ldh;
  if (R>8) { LaunchPrefillPle(a,nv1.p,g1.p); LaunchPrefillPle(b,nv2.p,g2.p); }
  else { PleMixKernel<<<4,512>>>(a); PleMixKernel<<<4,512>>>(b); }
  CheckLaunch(); Equal(h.Get(),Tight(p,R,ldh)); Equal(ring1.Get(),ring2.Get());
  Buffer<float> partial(Random((size_t)R*kHidden,.3f,67)),post(Random(R*4,.2f,71));
  Buffer<__half> svh(Half(Random(kHidden,.1f,73)));
  LinJob job{}; job.O=kHidden; job.R=R; job.S=1; job.partial=partial.p;
  job.svh=svh.p; job.aux=post.p; job.epi=kEpiResidual; job.dst=h.p;
  Buffer<LinJob> j1(std::vector<LinJob>{job}); job.dst=p.p; job.ldd=ldh;
  Buffer<LinJob> j2(std::vector<LinJob>{job}); std::vector<XfItem> items;
  for (int r=0;r<R;++r) for (int blk=0;blk<20;++blk) items.push_back({0,r,blk});
  Buffer<XfItem> it(items);
  LinOutKernel<<<(items.size()+7)/8,256>>>(j1.p,it.p,items.size());
  LinOutKernel<<<(items.size()+7)/8,256>>>(j2.p,it.p,items.size());
  CheckLaunch(); Equal(h.Get(),Tight(p,R,ldh));
  Buffer<float> packed((size_t)R*kWide);
  HIP_OK(hipMemcpy2DAsync(packed.p,kWide*sizeof(float),p.p,ldh*sizeof(float),
                         kWide*sizeof(float),R,hipMemcpyDeviceToDevice,0));
  CheckLaunch(); Equal(h.Get(),packed.Get());
  std::printf("PASS: storage rows %d\n",R);
}
static void CheckFused(int R,int K,int rm) {
  constexpr int I=128,O=128,ldh=kWide+64;
  const auto values=Random((size_t)R*kWide,.3f,79);
  std::vector<float> padded((size_t)R*ldh,123.f);
  for (int r=0;r<R;++r) std::copy_n(values.data()+(size_t)r*kWide,kWide,padded.data()+(size_t)r*ldh);
  Buffer<float> h(values),p(padded),x(Random((size_t)R*I,.3f,83)),post(Random(R*4,.2f,89));
  std::mt19937 random(97); std::vector<uint32_t> raw((I/16)*(O/16)*8*K);
  for (auto& v:raw) v=random();
  Buffer<uint32_t> tiles(raw);
  Buffer<__half> suh(std::vector<__half>(I,__float2half(1.f))),svh(std::vector<__half>(O,__float2half(1.f)));
  Buffer<float> partial((size_t)R*O); Buffer<__half> xh((size_t)R*I);
  LinJob job{}; job.tiles=tiles.p; job.suh=suh.p; job.svh=svh.p; job.src=x.p; job.dst=h.p;
  job.aux=post.p; job.I=I; job.O=O; job.K=K; job.R=R; job.lds=I; job.epi=kEpiResidual;
  PrefillLinear first,second;
  first.Build({job},{rm,true},nullptr,partial.p,partial.n,xh.p,xh.n);
  job.dst=p.p; job.ldd=ldh;
  second.Build({job},{rm,true},nullptr,partial.p,partial.n,xh.p,xh.n);
  if (!first.fused || !second.fused) std::abort();
  first.Run(); second.Run(); CheckLaunch(); Equal(h.Get(),Tight(p,R,ldh));
  first.Free(); second.Free();
  std::printf("PASS: fused residual rows %d K%d rm%d\n",R,K,rm);
}
int main() {
  for (int r : {9,24,33,64,65,513,1024}) Check(r);
  for (int r : {1,8,9,24,33,1024}) CheckStorage(r);
  for (int r : {9,33}) for (int k : {3,6}) for (int rm : {2,8}) CheckFused(r,k,rm);
}
