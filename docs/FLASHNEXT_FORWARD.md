# Qwen3.8-Flash-Next forward specification

## Scope and source identities

This is the text trunk and MTP predictor specification for Crucible on AMD gfx1151, not a vision encoder or sampler specification. Architecture spellings are HF `qwen4_exp` and GGUF `qwen4exp`. Mathematical semantics follow the pinned HF/PyTorch trunk. MTP follows the independently agreeing TensorFold and Gufo implementations. Numerical compatibility is a separate choice: the three engines do **not** produce identical floating-point results.

Sources were read at these identities. Citation prefixes below denote files relative to these roots:

| Prefix | Root and identity |
| --- | --- |
| `E/` | `exllamav3/` in the ExLlamaV3 repository, ExLlamaV3 `d3739fd393337b1ff4d6c2a342b12f0c87a9592f` |
| `T/` | `src/tensorfold/` in the TensorFold repository, TensorFold `17c73e189f5e6a5304cda7ea37f086f9c49b4788` |
| `G/` | `gufo/src/models/qwen38_flash_next/`, Gufo `373f79d64c29cb0e06c22390cf1c9d1f70fd7e2f` |
| `HF` | Transformers `3713bd839e580d07e4b70f2c89e986cb3c0e8ddf`, `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py` |
| `HFC` | same Transformers revision, `configuration_qwen4_exp.py` |
| `C` | `config.json` of the checkpoint below |
| `Q` | `quantization_config.json` of the checkpoint below |
| `I` | a normalized inventory of the safetensors headers of the checkpoint below, made during this audit (not in this repository) |

Checkpoint: [`turboderp/Qwen3.8-Flash-Next-exl3`, branch `3.05bpw_h5_ng5`](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.05bpw_h5_ng5), resolved to **`69e33439ae950f17bcbe95c98f117d80f759ab6d`**. Config and quantization manifest were downloaded from that branch, then their tensor geometry was checked against headers at the resolved immutable revision. EXL3 version is 1.4.4. `I` records original file, shape, dtype, header length and byte offsets, not inferred tensor decoding. The audit used fresh read-only source clones.

## 1. Geometry, conventions and numerical contract

For one unpadded text sequence, `t` is the absolute token position, `D=2560`, `S=4`, `W=S*D=10240`, low rank `L=320`, vocabulary `V=248320`. Linear notation is `linear(A,x)=A x`, with checkpoint matrix `[out,in]`, no bias unless explicitly mentioned. `sigmoid(x)=1/(1+exp(-x))`, `SiLU(x)=x*sigmoid(x)`. All norm epsilons here are **1e-6**. No dropout is applied during inference. All 48 trunk blocks contain MoE. Zero-based layers `3,7,...,47` use QSA, giving 12 QSA and 36 DeltaNet blocks. PLE is injected into zero-based layer **1**, not layer 2: `ple_layer_ids=[2]` is one-based. [C:16-155; E/architecture/qwen4_exp.py:85-215,254-284; G/config.cpp:96-177; HF:1258-1271]

Define centered RMSNorm over a group of `n` features:

```text
CN(x,w,n) = cast_input_dtype(x_fp32 * rsqrt(sum(x_fp32^2)/n + 1e-6)
                             * (1 + w_fp32))
```

The `+1` is in fp32. The pack stores **gamma minus one** for hyper-connection, attention q/k, indexer q/k, PLE and MTP input norms. GDN gated RMSNorm instead stores direct gamma. Grouped norms normalize each 2560-feature stream independently but use distinct gamma entries for all 10240 channels. [HF:152-195; T/families/qwen4_exp/model_layers.py:121-157; T/families/qwen4_exp/cuda/exl3.py:140-173]

**HF-compatible dtype contract:** the original model is bf16. Keep bf16 at model-operation boundaries, fp32 norm statistics, fp32 QSA scores, fp32 softmax evaluation, and fp32 DeltaNet state/decay/recurrence. GDN q/k normalization is fp32, not a bf16 RMSNorm. Do not infer activation dtype from the EXL3 pack's fp16 storage. ExLlamaV3 deliberately uses fp32 residual streams and fp16 sublayer inputs/intermediates instead. Gufo GGUF also has its own fp32/fp16 and activation-quantization schedule. TensorFold's row-invariant bf16 schedule is specified in section 4; it is not a claim of bitwise equivalence to every HF backend. [HF:167-195,403-459,576-610,811; E/architecture/qwen4_exp.py:119-198; E/modules/hyperconnections.py:26-39,219-244; G/reference.cpp:26-67; T/families/qwen4_exp/cuda/glue.py:1-21]

### State per sequence

- Each DeltaNet block: raw pre-conv projection history `[3,10240]`, zero initially; state `[48,128,128]` fp32, zero initially. ExLlamaV3 can retain four conv slots for its update API; only three past projections are mathematically needed.
- PLE: last two literal token IDs, EOS-padded initially; normalized gated-value history `[9,10240]`, zero initially.
- Each QSA block: K and V `[t+1,2,256]`, raw indexer K `[t+1,128]`, completed pooled indexer K `[floor((t+1)/4),128]`. Plain bf16 is the compatibility baseline. KV quantization is an optional engine approximation, not this checkpoint's architecture.
- MTP has its **own** QSA KV/indexer caches and predictor position. It does not reuse trunk QSA cache entries.

[HF:432-443,485-493,536-566,1133-1160,1218-1233; G/reference.cpp:70-103; T/families/qwen4_exp/model_layers.py:211-286; T/families/qwen4_exp/cuda/forward.py:209-215]

## 2. One decode token, in execution order

### 2.1 Embedding and stream expansion

Gather `e=embed_tokens[token]`, shape `[2560]`, stored bf16. Tile **whole vectors**, not individual channels: `H[s,d]=e[d]`, flattened stream-major as `[e,e,e,e]`. There is no embedding scale. Preserve the final pre-collapse `H` for MTP. [Q:15-27; HF:1423-1424,1478-1495; E/modules/hyperconnections.py:26-39; G/reference.cpp:559-575]

For each trunk block `i=0..47`, perform optional PLE, attention hyper-connection read, attention branch, hyper-connection write-back, MoE hyper-connection read, MoE branch, write-back. [HF:1273-1310; G/reference.cpp:578-597; T/families/qwen4_exp/model.py:229-250]

### 2.2 PLE, only before layer 1's attention mixer

#### Exact hash and gather

Use literal token IDs, including multimodal placeholder IDs if embeddings have been substituted. Hashing does not use embedding aliases. Let `u0=token[t]`, `u1=token[t-1]`, `u2=token[t-2]`. Missing history is `EOS=248044`. For shifts 1 and 2, substitute EOS if the source is before sequence start or before the segment following the most recent **previous** EOS. The current token being EOS does not erase its own older context. An older EOS cuts all context on its left. Only hash history resets at EOS; neither PLE convolution nor DeltaNet state is automatically zeroed by this hash rule. [HF:1117-1160; G/ngram.cpp:26-66; E/modules/ple.py:435-455]

Actual checkpoint multipliers, read from `ngram_embedding.safetensors` auxiliary tensors:

```text
m = [23703573157769, 20109073645365, 8052911324071]
hash2 = (u0*m0) XOR (u1*m1)
hash3 = hash2 XOR (u2*m2)
row[h] = (hash2 if h < 8 else hash3) % vocab[h] + offset[h]
```

Products/XOR are 64-bit integer operations. These checkpoint constants keep valid vocabulary-token products nonnegative and below signed 2^63, so signed PyTorch remainder and unsigned C++ modulo agree. Do not use a 32-bit multiply. [HF:1048-1058,1160-1176; G/ngram.cpp:31-55; T/families/qwen4_exp/model.py:41-82,115-142; checkpoint auxiliary byte range `26456..26735`, interpreted as 35 little-endian int64 values]

| Head | Order | Modulus | Offset |
| ---: | ---: | ---: | ---: |
| 0 | 2 | 20000003 | 0 |
| 1 | 2 | 20000023 | 20000003 |
| 2 | 2 | 20000033 | 40000026 |
| 3 | 2 | 20000047 | 60000059 |
| 4 | 2 | 20000059 | 80000106 |
| 5 | 2 | 20000063 | 100000165 |
| 6 | 2 | 20000069 | 120000228 |
| 7 | 2 | 20000077 | 140000297 |
| 8 | 3 | 20000081 | 160000374 |
| 9 | 3 | 20000093 | 180000455 |
| 10 | 3 | 20000107 | 200000548 |
| 11 | 3 | 20000147 | 220000655 |
| 12 | 3 | 20000153 | 240000802 |
| 13 | 3 | 20000159 | 260000955 |
| 14 | 3 | 20000161 | 280001114 |
| 15 | 3 | 20000171 | 300001275 |

The heads occupy disjoint table regions. Total unpadded rows are 320001446, padded to **320001536**, split into 128 shards of 2500012 rows. Each row has 160 features; concatenate 16 looked-up rows in head order to `[2560]`. Moduli are the first 16 primes strictly after 19999999. Multiplier derivation uses seed 1234, PLE index 0, SplitMix64 constants `0x9E3779B97F4A7C15`, `0xBF58476D1CE4E5B9`, `0x94D049BB133111EB`, and `base_seed=1234+10007*ple_index`. Always validate derived constants against stored auxiliaries. [HFC:156; HF:1034-1058,1080-1116; T/families/qwen4_exp/cuda/exl3.py:175-183; Q:54129-55185]

The 5-bit PLE `.trellis` is a **row codec**, not an EXL3 linear matrix. Each row is 51 int16 words: one fp16 scale plus 800 packed bits. Dequantization returns fp16 `codebook_value*row_scale + head_bias[h,d]`, where head bias is `[16,160]` fp16. TensorFold converts these looked-up values to its bf16 activation path. Keep the table out of the linear-weight streamer. [E/modules/ngram_embedding.py:16-23,188-206; T/families/qwen4_exp/cuda/exl3_pack.py:110-156; TensorFold root-relative `tests/cuda/test_qwen4_exp_exl3.py:60-79`; Q:54129-55185]

#### PLE forward

With `Eple[2560]` the gathered vector and input streams `H[4,2560]`:

```text
K = CN(reshape(linear(key_proj,Eple), [4,2560]), norm_key, 2560)
V = linear(value_proj,Eple)                                  # [2560]
Q = CN(H, norm_query, 2560)                                  # each stream
score[s] = sum_d K[s,d]*Q[s,d] / sqrt(2560)
a[s] = sign(score[s])*sqrt(max(abs(score[s]), 1e-6))
U[s,d] = sigmoid(a[s])*V[d]                                  # [4,2560]
N = CN(U, norm_conv, 2560)
conv[t,c] = sum_{j=0..3} conv_weight[c,j]*N[t-9+3*j,c]
PLE[t] = U[t] + SiLU(conv[t])
H[t] = H[t] + PLE[t]
```

`key_proj=[10240,2560]`, `value_proj=[2560,2560]`, both plain fp16 in this pack. PLE conv is depthwise `[10240,1,4]`, no bias, dilation **3**, causal zero padding, and keeps the last **9 normalized input rows**, not the conv output. Norm weights are bf16; conv is fp16 in this pack. The sign at exactly zero is zero, yielding sigmoid gate 0.5. HF uses bf16 products/sums at its operation boundaries; ExLlamaV3 computes PLE key/query and gated values in fp32 and conv input in fp16. [HF:1183-1255; E/modules/ple.py:155-214,344-358,418-433; G/reference.cpp:151-222; Q:55186-55242; I:80459-80473]

### 2.3 Hyper-connection read and write around each sublayer

This is **gated residual**, not DeepSeek manifold HC: no Sinkhorn and no 4x4 stream-combine matrix. Given raw streams `H`:

```text
N[s,d] = CN(H[s,d], hc_norm[s,d], 2560)
f = flatten_stream_major(N)                                 # [10240]
a = SiLU(linear(input_mix_weight_down,f)/4)                  # [320]
b = sigmoid(linear(input_mix_weight_up,a))                  # [10240]
x[d] = (1/4)*sum_s reshape(b,[4,2560])[s,d]*N[s,d]           # [2560]
inject[s] = 2*sigmoid(linear(block_inject_weight,f)[s]/4)     # [4]
y = sublayer(x)                                             # [2560]
Hnew[s,d] = H[s,d] + inject[s]*y[d]
```

Down/up shapes are `[320,10240]` / `[10240,320]`; injection matrix is `[4,10240]`. All three are plain fp16 weights. The mixer reads **normalized** streams, but write-back adds to the **raw** input streams. Injection is one scalar per stream, not per channel, while input mixing is per channel. Norm gamma has separate per-stream entries. Attention and MoE have independent mixer weights. Divisions by 4 occur after down and injection projections and after the stream sum, not after up. [HF:1003-1031,1299-1310; E/modules/hyperconnections.py:219-244,403-415,468-494; T/families/qwen4_exp/model_layers.py:160-200; G/reference.cpp:106-146; Q:30-64]

### 2.4 Gated DeltaNet branch, layers other than `3 mod 4`

Projection geometry and order:

| Projection | Weight shape | Output / layout |
| --- | --- | --- |
| `in_proj_qkv` | `[10240,2560]` | contiguous `[q:2048 | k:2048 | v:6144]` |
| `in_proj_z` | `[6144,2560]` | `[48,128]` output gate logits |
| `in_proj_b` | `[48,2560]` | beta logits |
| `in_proj_a` | `[48,2560]` | decay/time-step logits |
| `out_proj` | `[2560,6144]` | branch output `[2560]` |

A fused runtime projection can concatenate `[qkv | z | b | a]`, width **16480**. That does not change checkpoint qkv splitting, and z/a/b do **not** pass through conv. [HF:465-509,527-532,565-585; T/families/qwen4_exp/cuda/gdn.cu:57-64; T/families/qwen4_exp/cuda/exl3.py:166-173; Q:65-193]

1. Project x to qkv, z, b, a.
2. Depthwise causal conv over the raw qkv projection, kernel 4, no bias:
   `c[t,ch]=SiLU(sum_{j=0..3} wconv[ch,j]*qkv[t-3+j,ch])`.
   Save raw qkv, not SiLU output, in the conv history. Split c into `q[16,128]`, `k[16,128]`, `v[48,128]`.
3. Normalize in fp32: `qhat=q/sqrt(sum(q*q)+1e-6)/sqrt(128)` and `khat=k/sqrt(sum(k*k)+1e-6)`.
4. For value head h, use key/query head **floor(h/3)**. Native HF repeats each key head three adjacent times. This is not h modulo 16 unless weights/state have been permuted to the GGUF tiled layout.
5. Compute in fp32 `glog[h]=-exp(A_log[h])*softplus(a[h]+dt_bias[h])`, `decay[h]=exp(glog[h])`; softplus uses the usual threshold 20 branch. `beta[h]=sigmoid(b[h])`, rounded to activation dtype before the recurrence. A_log and dt_bias are `[48]` bf16 stored values, widened to fp32. A_log is a logarithm, not the already-negated A used by GGUF.
6. Update the fp32 state and read the current token from the **updated** state.

For state orientation `[key,value]`, per value head:

```text
Sd = decay*Sprevious                                       # [128,128], fp32
memory[j] = sum_i Sd[i,j]*khat[i]
delta[j] = beta*(v[j]-memory[j])
Snew[i,j] = Sd[i,j] + khat[i]*delta[j]
y[j] = sum_i Snew[i,j]*qhat[i]
```

An implementation may transpose the state to `[value,key]`, as TensorFold/Gufo do, but must transpose the contractions too. Initial state is zero. Keep both the state and these contractions fp32. [HF:403-459,534-621; E/modules/gated_delta_net.py:1138-1209; G/reference.cpp:242-322; T/families/qwen4_exp/cuda/gdn.cu:31-45,57-140]

7. Per-head gated RMSNorm over 128 values, direct gamma `[128]` shared across all 48 heads, with **sigmoid z**, not SiLU z, and norm **before** gate. HF's precise boundary order is:

```text
yb = bf16(y)
r = rsqrt(mean(fp32(yb)^2)+1e-6)
yn = bf16(bf16(fp32(yb)*r)*gamma)
yg = bf16(fp32(yn)*sigmoid(fp32(z)))
```

Then concatenate `[48,128]` to `[6144]`, apply out_proj, and perform the attention-site HC write-back. The gamma multiplication is in input dtype before the fp32 gate; this differs from centered RMSNorm's single final cast. [HF:180-195,616-624; T/families/qwen4_exp/cuda/gdn.cu:115-140; E/architecture/qwen4_exp.py:130-136]

### 2.5 QSA sparse attention branch, layers `3 mod 4`

Project x with the following native checkpoint layout:

| Projection | Weight shape | Output |
| --- | --- | --- |
| `q_proj` | `[12288,2560]` | reshape `[24,512]`, each head `[q:256 | gate:256]` |
| `k_proj` | `[512,2560]` | `[2,256]` |
| `v_proj` | `[512,2560]` | `[2,256]` |
| `indexer.index_qk_proj` | `[640,2560]` | `[4*128 query | 1*128 raw key]` |
| `o_proj` | `[2560,6144]` | `[2560]` |

The query gate is interleaved **per head**, not one global `[6144 q | 6144 gate]`. Apply CN to q and k independently per 256-feature head, same gamma shared across heads. V has no norm or RoPE. Indexer query heads have their own CN over 128; raw indexer keys are cached **before** norm or RoPE. [HF:673-718,819-900; E/architecture/qwen4_exp.py:144-196; G/reference.cpp:336-393; T/families/qwen4_exp/model_layers.py:337-364,420-454]

#### Partial, interleaved-section RoPE

Rotary dimension is **64**, even on 128-dimensional indexer heads. Rotate only channels 0..63; leave the rest untouched. This is split-half pairing, `(j,j+32)` for j=0..31, not adjacent-channel pairing. `inv_freq[j]=10000000^(-2*j/64)`. Text has identical temporal/height/width positions, all equal to t. For multimodal coordinates `p=(pT,pH,pW)`, section sizes are **[11,11,10]** with interleaved mRoPE:

```text
axis(j) = H if j % 3 == 1 and j < 33
          W if j % 3 == 2 and j < 30
          T otherwise
angle[j] = p[axis(j)]*inv_freq[j]
rot[j]    = x[j]*cos(angle[j]) - x[j+32]*sin(angle[j])
rot[j+32] = x[j+32]*cos(angle[j]) + x[j]*sin(angle[j])
```

HF computes frequencies/cos/sin fp32 then casts cos/sin to x dtype. TensorFold CUDA evaluates its rotation in fp32 and rounds the rotated output to bf16. Apply to Q, K and normalized indexer Q at the query token's position; pooled indexer K uses the **block-start** position. [HF:79-149,629-669,713-745; G/reference.cpp:365-393; T/families/qwen4_exp/cuda/glue.py:276-338; T/families/qwen4_exp/cuda/attention.py:192-225; C:137-146]

#### Pool, score, select

At query t, let `n=t+1`, `b=floor(n/4)`. Cache current main K/V and raw indexer K. On completing a four-token block j:

```text
raw_pool[j] = fp32_mean(raw_indexer_K[4*j : 4*j+4])
pooled_K[j] = RoPE(CN(cast_activation_dtype(raw_pool[j]), indexer_k_norm,128),
                  position_of_token(4*j), rotary_dim=64)
score[j] = sum_{h=0..3} max(dot_fp32(indexer_Q[h],pooled_K[j]),0)/sqrt(128)
```

Only complete causal blocks j<b are selectable. If b<=512, attend to all n tokens. Otherwise select the best **512 blocks**, expand to **2048 token positions**, and append the incomplete tail `4*b..t`, length 0..3. Total attended tokens can therefore be **2051**, not always 2048. There is **no fixed recent sliding window**. The unfinished tail is always present, but once that block completes it competes like every other block. At n divisible by 4, even the current token's complete block is not guaranteed selected. [HF:728-765; E/modules/qsa_indexer.py:147-162,199-239; G/reference.cpp:404-484; T/families/qwen4_exp/cuda/attention.py:228-291]

**Crucible deterministic selection rule:** compare score descending, block ID ascending on equal fp32 score. Emit chosen blocks in ascending block-ID order, tokens ascending inside blocks, tail last. TensorFold CUDA implements this rule; Gufo's scalar stable sort implements equivalent tie selection. ExLlamaV3 optimized DSA top-k uses score-desc/index-asc. **HF `torch.topk` itself does not promise stable ties**, and ExLlamaV3's torch fallback and TensorFold's unfused MLX `argpartition` inherit backend tie behavior. Stable ties are an intentional deterministic refinement, not a PyTorch bitwise guarantee. [T/families/qwen4_exp/cuda/attention.py:250-291,303-353; E/modules/qsa_indexer.py:226-239,283-288; G/reference.cpp:455-473; HF:757]

For padded/packed inputs HF forms blocks from the ordered **visible token list**, not necessarily physical absolute indices `4*j`. The absolute-index formula above is valid only for contiguous unpadded sequence caches. Do not mix requests or reset segments inside one physical block without adapting visibility. [HF:727-746]

#### Main GQA and output gate

For query head h, KV head is **floor(h/12)**. Over selected positions P:

```text
logit[p] = dot(q[h],K[p,floor(h/12)]) / sqrt(256)             # scale 1/16
prob = softmax_fp32(logit), then cast to activation dtype
context[h] = sum_p prob[p]*V[p,floor(h/12)]
gated[h,d] = context[h,d]*sigmoid(query_gate[h,d])
y = linear(o_proj, flatten_heads(gated))                     # [2560]
```

Then attention-site HC write-back. No attention-logit softcap, no sink token, no sliding-window bias, no indexer gate projection beyond its q/k projection. Sparse selection masks the ordinary causal attention; it does not replace K/V with pooled features. [HF:782-817,847-900; G/reference.cpp:486-526; T/families/qwen4_exp/model_layers.py:453-478]

### 2.6 MoE branch, every layer

Use the MoE HC read to obtain x `[2560]`. Router weight `[512,2560]`; compute 512 logits. Reference activation is **softmax**, not sigmoid. Take **top 10**, then renormalize selected probabilities to sum to one. Default `norm_topk_prob=True` comes from HF config even though the checkpoint config does not explicitly set it. No router bias, correction bias, group selection or routed scaling factor is configured, so routed scale is **1**. [HF:960-979; HFC:163; E/architecture/qwen4_exp.py:200-229; T/families/qwen4_exp/model_layers.py:494-527; G/reference.cpp:530-548]

For selected expert e, with intermediate width 640:

```text
a_e = SiLU(linear(gate_proj[e],x))*linear(up_proj[e],x)       # [640]
y_e = linear(down_proj[e],a_e)                               # [2560]
y_routed = sum_{selected e} weight[e]*y_e
```

Gate/up matrices `[640,2560]`, down `[2560,640]`. There are 512 experts per layer. Shared expert is a separate 640-wide SwiGLU with the same three matrix shapes, always evaluated. Its scalar gate is `sigmoid(linear(shared_expert_gate,x))`, weight `[1,2560]`. Final branch is `y_routed + shared_gate*y_shared`, followed by the MoE-site HC write-back. Shared expert is not part of the top-10 softmax; TensorFold runtime's pseudo-expert ID 512 is only an implementation convention. [HF:904-958,981-1000; G/reference.cpp:550-590; T/families/qwen4_exp/cuda/exl3.py:154-157; T/cuda/moe.py:57-89]

For Crucible's deterministic schedule, route by descending **fp32 logits**, lower expert ID on equal logits. Selected softmax weights can be computed directly as `exp(l_e-max_selected)/sum_selected exp(...)`, mathematically identical to full-softmax top-k followed by renormalization. Combine top-10 slots in selection order, then shared expert as slot 10, accumulating in fp32 and rounding the combined branch once. HF eager instead accumulates selected experts by expert ID into bf16 with `index_add_`, so its rounding differs; see section 4. [T/cuda/moe.py:57-89,156-179; HF:940-958; G/kernels/rocm/kernels.hip.cpp:2420-2494]

### 2.7 Final mixer, LM head and logits

After layer 47 MoE write-back, retain the raw `[4,2560]` stack. Apply a dedicated final HC mixer with the same grouped centered norm and down/up/read formulas as section 2.3, but **no injection projection, no branch and no residual write-back**. Its result is `[2560]`. There is **no separate final model RMSNorm**. Apply untied `lm_head[248320,2560]` to obtain logits `[248320]`. No logit softcap or output rescaling is present. Softmax/sampling is outside the model forward. [HF:1390-1395,1493-1495,1664,2515; E/architecture/qwen4_exp.py:277-301; G/reference.cpp:599-607; Q:2598844-2598905]

## 3. Prefill, verification windows and MTP

### 3.1 Prefill differences

The same equations apply to `[B,R,D]` rows. Stream tiling is `[B,R,4,D]`; HC and router work independently per row. Only conv/recurrence and causal caches couple rows:

- GDN conv can run as a batched causal conv over `[old_tail; projections]`. Prefill HF uses chunked gated-delta computation, whereas one-token decode uses recurrent computation. A chunked WY/scan kernel must implement the same recurrence, initialize from the carried fp32 state, and save the final state. It need not be bit-identical to serial recurrence unless its arithmetic is deliberately fixed.
- PLE hashes use preceding history and current rows; its conv keeps the last nine normalized rows across chunk boundaries. EOS masking changes hashes, not conv history.
- QSA pools only complete blocks. Query r can select only blocks completed by position `past+r`; do not use future keys just because the whole chunk was projected. The final partially pooled block must continue correctly across chunks.
- All intermediate token rows are required to update states/caches, but normal generation can evaluate final mixer/LM head only for the last prompt row. Teacher-forced scoring requests all rows.

[HF:536-566,587-610,727-765,1133-1160,1218-1233; T/families/qwen4_exp/cuda/forward.py:170-192,239-255,400-424; G/reference.cpp:248-273; TensorFold recipe `recipes/qwen3.8-flash-next.md:30-58` in the TensorFold documentation]

TensorFold's EXL3 prompt path dequantizes dense matrices once per prompt chunk and uses fixed-tile GEMM; grouped routed experts run in windows up to 1024 rows. This is a runtime throughput choice, not a change to token equations. ExLlamaV3 uses different HC kernels at small versus large row counts, including its tiled int8 HC path on Ampere+ NVIDIA, not HIP. Gufo reports scalar/bulk prefill are not bit-identical. Do not assume matching cache state across engines merely from matching token counts. [T/families/qwen4_exp/cuda/exl3_pack.py:16; E/modules/hyperconnections.py:244-253,332-350,415-465; Gufo `gufo/docs/models/qwen3.8-flash-next/QUALITY.md:12-23`; TensorFold recipe:158-195]

### 3.2 Multi-row verification

A verification window of R consecutive tokens is mathematically R serial causal steps, not R independent requests. Each row needs its own HC gates, route, QSA selected list and logits. DeltaNet must process rows in order using previous row state; causal conv similarly crosses window rows. Pooling and visibility use `end=past+r+1`. Truly independent requests have separate recurrent/conv/hash/KV state even when projections and experts share a launch. [T/families/qwen4_exp/cuda/gdn.cu:57-153; T/families/qwen4_exp/cuda/attention.py:228-265; T/families/qwen4_exp/cuda/forward.py:135-168,442-490]

Row-exact speculation requires:

1. Shape-only tiling and split-K, identical reduction/contraction order for 1-row and R-row projections, routers and experts.
2. Identical recurrent update code in forward and replay. TensorFold CUDA compiles the update without FMA contraction and shares it with prefix replay.
3. On accepting k<R rows, commit only prefix k: restore/replay each GDN state, retain last three raw qkv rows and last nine normalized PLE rows after k, advance hash history only by accepted tokens, truncate main/indexer KV lengths and discard pooled blocks no longer complete. Preserve the raw incomplete-block keys for future completion.
4. MTP's cache and recursive hidden carry are also speculative and must be realigned to the accepted target prefix. A stale draft prefix must not become committed trunk state.

[T/families/qwen4_exp/cuda/gdn.cu:3-8,31-45,155-180; T/families/qwen4_exp/cuda/forward.py:563-596; T/families/qwen4_exp/model_layers.py:239-246; E/modules/gated_delta_net.py:244-316; T/families/qwen4_exp/runtime.py:28-43; TensorFold recipe:30-36]

### 3.3 MTP predictor specification

Let trunk input token at t be `x_t`, its **pre-final-mixer** residual stack be `H_t[4,2560]`, and its sampled next token be `x_(t+1)`. MTP takes **H_t plus the embedding of x_(t+1)** and predicts x_(t+2). Do not feed the collapsed `[2560]` trunk hidden, pre-attention hidden, or original input embedding. `mtp_use_hidden_state_from_layer=null` means the last trunk stack for this checkpoint. [E/architecture/qwen4_exp_mtp.py:18-36,133-167; T/families/qwen4_exp/mtp.py:1-47; C:94-109]

Input fusion:

```text
E = CN(embed_tokens[x_(t+1)], pre_fc_norm_embedding, 2560)
N = CN(flatten(H_t), pre_fc_norm_hidden, 10240)               # ONE full-width norm
Ep = linear(fc_embedding,E)                                 # [2560]
Hp[s] = linear(fc_hidden,reshape(N,[4,2560])[s])             # shared [2560,2560]
F[s,d] = Hp[s,d] + Ep[d]                                    # no averaging/broadcast of Hp
```

Both FC weights are shared across four streams. The full-width hidden norm is **not grouped**; its gamma is `[10240]`. This is a key difference from every HC norm. [T/families/qwen4_exp/mtp.py:20-47; G/reference.cpp:634-671; G/kernels/rocm/executor.cpp:2556-2574]

Then execute one dedicated decoder block: attention HC read, full **QSA** attention with its own cache, HC write-back, MoE HC read, 512-expert/top-10 MoE plus shared expert, HC write-back. No PLE or GDN occurs in this checkpoint's MTP block. Its QSA dimensions/ratio/budget/RoPE equal the trunk QSA's. Retain output raw stack `Hdraft`, collapse with the MTP's own combine-less HC mixer, apply shared trunk LM head. Trunk and MTP share **only token embedding and LM head** here; FCs, norms, decoder block, final mixer and caches are dedicated. [E/architecture/qwen4_exp_mtp.py:79-115,143-162; T/families/qwen4_exp/mtp.py:20-47; T/families/qwen4_exp/cuda/exl3.py:211-217; G/reference.cpp:673-698]

To chain another draft, pair `Hdraft` with the embedding of the just-proposed token and reuse the same one-layer predictor, advancing its own attention position. Prompt catch-up feeds aligned pairs `(trunk_stack[t], next_token[t+1])`. ExLlamaV3 and TensorFold explicitly export raw stack, not final mixer output, for this chaining. [V:311-320,345-353; T/families/qwen4_exp/cuda/mtp.py:1,37-74; G/reference.cpp:629-630,686-698; E/architecture/qwen4_exp_mtp.py:29-58,184-192]

The EXL3 pack carries MTP final mixer in **`mtp_hyper_connection_mixer_patch.safetensors` outside the main index**, so load it explicitly. TensorFold optionally uses a restricted-vocabulary 4-bit requantized draft head; that is a draft proposal optimization, not the shared full-head mathematical predictor. Full trunk verification must still use the original full EXL3 LM head. [T/families/qwen4_exp/cuda/exl3_pack.py:15,38-47; T/families/qwen4_exp/cuda/exl3.py:225-231; T/families/qwen4_exp/cuda/mtp.py:55-68; TensorFold recipe:178-184]

## 4. Cross-check verdicts and exactness rules

### Semantic agreements and disagreements

| Topic | Verdict and implementation consequence |
| --- | --- |
| Trunk op order, HC equations, PLE layer, sigmoid GDN gate, QSA budget, MoE | All three agree mathematically with HF. The engine-specific storage layouts and round points are not interchangeable. See section 2's independent citations. |
| MTP hidden normalization | **ExLlamaV3 differs semantically.** It normalizes each stream over 2560, while TensorFold and Gufo normalize the complete 10240 vector. Two independent implementations confirm full-width normalization. HF ignores MTP weights, so there is no HF MTP-forward oracle. ExLlamaV3 marks its handling of this step as unverified in `qwen4_exp_mtp.py`. Choose full-width norm. [E/modules/arch_specific/qwen4_exp_mtp.py:12-24,145-150; T/families/qwen4_exp/mtp.py:24,44; G/reference.cpp:641-645; HF:1322-1324] |
| GDN head mapping and A parameter | Native HF/EXL3 use h//3 and A_log. Gufo uses h%16 with converted value-side tensors, and uses `exp(ssm_a*softplus(...))` where ssm_a represents negative exp(A_log). These are storage transformations, not permission to apply GGUF mapping to EXL3. [HF:580-584; G/reference.cpp:278-291; T/families/qwen4_exp/cuda/gdn.cu:59-60,102-105] |
| QSA score scale | Gufo's scalar reference (`373f79d`) does not divide by sqrt(128). This positive constant does not change exact top-k, but can alter rounding/ties if scores are rounded. Use HF's scaled fp32 score. [HF:755; G/reference.cpp:450-459; T/families/qwen4_exp/cuda/attention.py:241-245] |
| Top-k ties | TensorFold CUDA and optimized ExLlamaV3 impose lower-ID tie-breaking; Gufo scalar QSA stable-sort does too. HF topk and unfused MLX argpartition do not promise it. Use an explicit total order for row invariance. [HF:757,973; T/cuda/moe.py:57-81; T/families/qwen4_exp/model_layers.py:395,511; E/modules/qsa_indexer.py:283-288; G/reference.cpp:462-467] |
| Residual/norm activation dtypes | ExLlamaV3 carries fp32 streams, fp16 HC mixed inputs and expert intermediates; TensorFold (`17c73e1`) carries bf16 streams with op-boundary rounding. Gufo (`373f79d`) uses converted GGUF, fp32 residuals and fp16/quantized projection paths. HF bf16 streams are closer in dtype schedule to TensorFold, not ExLlamaV3. [E/modules/hyperconnections.py:26-39,415-465; E/architecture/qwen4_exp.py:135-140,222-243; T/families/qwen4_exp/cuda/glue.py:56-81; G/reference.cpp:26-67] |
| Router logits | TensorFold CUDA (`17c73e1`) keeps router dots fp32 before selection, a deliberate discrete-decision safeguard. HF's literal F.linear output is input dtype (normally bf16), then softmax is fp32. ExLlamaV3 uses fp16 router-logit buffers. Gufo's reference (`373f79d`) computes float with narrowed router weights. Mathematical softmax routing agrees, exact picks near ties may not. A true HF eager bitwise mode must reproduce its router cast, not silently substitute TensorFold's fp32 logits. [HF:969-977; T/cuda/moe.py:24-54,57-89; E/modules/block_sparse_mlp.py:751-762; G/reference.cpp:530-533] |
| MoE renormalization and combine | HF has no explicit denominator clamp here; Gufo (`373f79d`) clamps selected mass to fp16 min normal, 6.103515625e-5. For finite 512-way softmax, top-10 mass is at least 10/512, so the clamp does not affect normal valid inputs. HF eager uses expert-ID-order bf16 index_add; TensorFold uses selected-slot-order fp32 combine, round once. ExLlamaV3 also uses fp32 branch accumulation. Do not promise HF bit equality for the TF schedule. [HF:940-979; G/reference.cpp:541-548; T/cuda/moe.py:156-179; E/modules/block_sparse_mlp.py:1064-1073] |
| RoPE trig | HF casts fp32 cos/sin to input dtype before rotation; TF CUDA computes angle and trig in fp32 then rounds final rotated values. This is another arithmetic, not architecture, difference. [HF:124-140; T/families/qwen4_exp/cuda/glue.py:312-338] |

### TensorFold CUDA operation-boundary rounding

Let `B(x)` mean round-to-nearest-even bf16, then widen back to fp32 for an expression. These rules summarize the implemented schedule, including the EXL3 path. They are useful for deterministic Crucible kernels, but are not universal HF bit-exact guarantees.

- Centered norm: `B(x*rsqrt(mean(x*x)+eps)*gamma_fp32)`, one final rounding, no premature rounding of `1+w`.
- HC down/inject: projection outputs rounded to bf16; low activation `B(SiLU(B(down/4)))`; injection `B(2*B(sigmoid(B(inject_logit/4))))`. Up projection rounded to bf16. Read-out is `B(sum_s B(B(sigmoid(up[s]))*N[s])/4)` with stream sum in ascending order.
- HC write-back: combine routed slots in fixed order in fp32, round branch once, then `B(H+B(branch*inject))` per stream. Do not fuse through either bf16 boundary.
- Router: bf16 inputs/weights, **fp32 logits**, top-10 ordered by fp32 logit/lower ID, selected softmax evaluated fp32, selected weights rounded bf16. Shared gate logit is first rounded bf16, sigmoid rounded bf16. No atomic accumulation dependent on row count.
- GDN: projection values bf16; conv is fp32 sum and SiLU with **one final bf16 cast** in the fused decode chain; q/k L2 norms, decay and state updates fp32; beta `B(sigmoid(b))`; readout B(y); gated norm `B(B(B(y)*r)*gamma)` then `B(yn*sigmoid(z_fp32))`. Replay uses the same update function and no FMA contraction.
- Attention: q/k/indexer RMSNorm bf16 output then RoPE bf16 output; raw pooled-key mean fp32 then bf16 before its norm. Indexer scores remain fp32. Softmax accumulators are fp32; attention probability fragments are bf16 for the value contraction; output and output gate round to bf16.
- PLE: normed key/query bf16; element products `B(key*query)` before their fp32 sum; scaled dot B; signed-root B; sigmoid B; gated value B. Conv normalizes to bf16, sums fp32, rounds conv before SiLU, then `B(gated+SiLU(conv))`, then `B(H+PLE)`. These PLE and GDN conv round points are deliberately different in the existing kernels.
- MTP: full-width hidden norm and embedding norm bf16, FC outputs bf16, input stream add bf16.

[T/families/qwen4_exp/cuda/glue.py:12-21,56-100,154-200,221-231,245-255,280-338,390-400,477-557,560-572; T/cuda/moe.py:24-89,156-179; T/families/qwen4_exp/cuda/gdn.cu:3-8,76-105,115-140; T/families/qwen4_exp/cuda/attention.py:18-27,201-225]

Fix GEMM K slices, reduction trees, expert output order and FMA policy by tensor shape, not verification row count. TensorFold has load-time window checks and disables drafting if multi-row logits differ from serial steps. Exactness within one engine does not establish agreement across engines: the recipe reports approximately 95.9% top-1 agreement with ExLlamaV3 on teacher-forced rows, with different self-agreement when ExLlamaV3 batch shape changes. [T/families/qwen4_exp/runtime.py:28-43; TensorFold recipe:30-36,196-231]

## 5. Per-token checkpoint-weight byte accounting

### Accounting definition and manifest limitation

These are **exact serialized payload bytes in the selected weight working set**, not measured HBM transactions. One full tensor read is counted for each always-used weight, plus only 10 routed experts per layer. Scale vectors and 4-byte mul1 constants are included. File headers/alignment, cached derived copies, activation traffic, KV/state traffic, codebooks and speculative bookkeeping are excluded. Real HBM reads can be lower from caching/reuse or higher from rereads/dequantization.

The EXL3 manifest is incomplete for this purpose: it has **no MTP entries**, omits all 36 DeltaNet A_log/dt_bias/conv tensors and the PLE conv weight. Therefore a complete MTP or trunk byte table cannot honestly be obtained from Q alone. The audit downloaded only safetensors headers via HTTP Range, plus small PLE auxiliaries and the 13 MB MTP mixer patch. All **296941 tensors listed in Q** matched the header shapes and byte counts exactly. Stored per-family widths were uniform across all trunk layers/experts, so expert identity does not change this pack's one-row byte total. Omitted required weights are explicitly included below, marked `header`. The manifest-only streamed trunk subtotal is **4612299620 bytes**, short by **3037952 bytes**. [Q:15-2598905; I:80311-80351,80459-80473,3860471-3860599,3955787-3955832; T/families/qwen4_exp/cuda/exl3_pack.py:15,38-47]

For an EXL3 linear with logical `[out,in]=[N,K]` and b bits, this pack stores:

```text
payload_bytes = K*N*b/8 + 2*K (suh) + 2*N (svh) + 4 (mul1)
```

All dimensions in this pack satisfy tile alignment. This formula is not the n-gram row codec. Every routed expert matrix is 3 bits, every shared expert matrix 5 bits, GDN qkv/z/out and main QSA q/k/v/o are 5 bits, indexer matrix 3 bits, LM head 5 bits. MTP fc_hidden/fc_embedding are **4 bits**, despite `mtp_bits=3` in config. Headers, not global bpw hints, determine per-tensor width. [Q:65-134,159-193,264-298,54024-54128,163476-163710,2598871-2598905; I:3860534-3860599]

### 5.1 Trunk: every streamed tensor family, one decode row

`Instances` includes active expert multiplicity. Shapes are logical `[out,in]` for linears; all bytes are integers. HC rows combine attention and MoE sites, 96 sites total.

| Tensor family | Shape | Stored format | Instances | Bytes each | Bytes/token | Evidence |
| --- | --- | --- | ---: | ---: | ---: | --- |
| HC down | `[320,10240]` | fp16 plain | 96 | 6553600 | 629145600 | Q:30-41 |
| HC up | `[10240,320]` | fp16 plain | 96 | 6553600 | 629145600 | Q:40-48 |
| HC inject | `[4,10240]` | fp16 plain | 96 | 81920 | 7864320 | Q:49-56 |
| HC norm | `[10240]` | fp16 plain | 96 | 20480 | 1966080 | Q:57-64 |
| GDN qkv | `[10240,2560]` | EXL3 5-bit | 36 | 16409604 | 590745744 | Q:65-99 |
| GDN z | `[6144,2560]` | EXL3 5-bit | 36 | 9847812 | 354521232 | Q:100-134 |
| GDN b | `[48,2560]` | fp16 plain | 36 | 245760 | 8847360 | Q:135-146 |
| GDN a | `[48,2560]` | fp16 plain | 36 | 245760 | 8847360 | Q:147-158 |
| GDN out | `[2560,6144]` | EXL3 5-bit | 36 | 9847812 | 354521232 | Q:159-193 |
| GDN conv | `[10240,1,4]` | bf16 plain, header | 36 | 81920 | 2949120 | I:80324-80338 |
| GDN A_log | `[48]` | bf16 plain, header | 36 | 96 | 3456 | I:80311-80323 |
| GDN dt_bias | `[48]` | bf16 plain, header | 36 | 96 | 3456 | I:80339-80351 |
| GDN direct-gamma norm | `[128]` | bf16 plain | 36 | 256 | 9216 | Q:194-204 |
| QSA q+gate | `[12288,2560]` | EXL3 5-bit | 12 | 19690500 | 236286000 | Q:163533-163567 |
| QSA k | `[512,2560]` | EXL3 5-bit | 12 | 825348 | 9904176 | Q:163568-163602 |
| QSA v | `[512,2560]` | EXL3 5-bit | 12 | 825348 | 9904176 | Q:163603-163637 |
| QSA o | `[2560,6144]` | EXL3 5-bit | 12 | 9847812 | 118173744 | Q:163638-163672 |
| QSA q/k norms | `[256]` each | bf16 plain | 24 | 512 | 12288 | Q:163673-163705 |
| QSA indexer q+k | `[640,2560]` | EXL3 3-bit | 12 | 620804 | 7449648 | Q:163476-163510 |
| QSA indexer q/k norms | `[128]` each | bf16 plain | 24 | 256 | 6144 | Q:163511-163532 |
| Router | `[512,2560]` | fp16 plain | 48 | 2621440 | 125829120 | Q:240-251 |
| Shared scalar gate | `[1,2560]` | fp16 plain | 48 | 5120 | 245760 | Q:252-263 |
| Routed expert up | `[640,2560]` | EXL3 3-bit | 480 | 620804 | 297985920 | Q:264-298 |
| Routed expert gate | `[640,2560]` | EXL3 3-bit | 480 | 620804 | 297985920 | Q:299-333 |
| Routed expert down | `[2560,640]` | EXL3 3-bit | 480 | 620804 | 297985920 | Q:334-368 |
| Shared expert up | `[640,2560]` | EXL3 5-bit | 48 | 1030404 | 49459392 | Q:54024-54058 |
| Shared expert gate | `[640,2560]` | EXL3 5-bit | 48 | 1030404 | 49459392 | Q:54059-54093 |
| Shared expert down | `[2560,640]` | EXL3 5-bit | 48 | 1030404 | 49459392 | Q:54094-54128 |
| PLE key | `[10240,2560]` | fp16 plain | 1 | 52428800 | 52428800 | Q:55186-55197 |
| PLE value | `[2560,2560]` | fp16 plain | 1 | 13107200 | 13107200 | Q:55198-55209 |
| PLE three norms | `[10240]` each | bf16 plain | 3 | 20480 | 61440 | Q:55210-55242 |
| PLE conv | `[10240,1,4]` | fp16 plain, header | 1 | 81920 | 81920 | I:80459-80473 |
| Final HC down/up | same as site | fp16 plain | 2 | 6553600 | 13107200 | Q:2598844-2598860 |
| Final HC norm | `[10240]` | fp16 plain | 1 | 20480 | 20480 | Q:2598861-2598870 |
| LM head | `[248320,2560]` | EXL3 5-bit | 1 | 397813764 | 397813764 | Q:2598871-2598905 |
| **Total streamed weights** | | | | | **4615337572** | sum above |

Category totals:

| Category | Bytes/token |
| --- | ---: |
| HC, including final mixer | 1281249280 |
| GDN, including plain state parameters | 1320448176 |
| QSA, including indexer/norms | 381736176 |
| Routers | 125829120 |
| Selected routed experts | 893957760 |
| Shared experts plus scalar gates | 148623936 |
| PLE projections/norms/conv, excluding gathers | 65679360 |
| Full LM head | 397813764 |
| **Total** | **4615337572 (4.298368 GiB)** |

Format totals: **3121655652 EXL3 bytes**, **1490636800 fp16-plain bytes**, **3045120 bf16-plain bytes**. The HC down/up weights alone cost 1.2714 GB across 97 mixers; this architecture is not an expert-only bandwidth problem.

### 5.2 Gathered, not streamed

| Gather | Bytes per token | Resident storage / notes |
| --- | ---: | --- |
| Main token embedding row | **5120** | `[2560]` bf16; full table 1271398400 bytes, do not stream it per token. Q:15-27 |
| PLE packed rows | **1632** | 16 disjoint heads times 102 bytes/row; table payload 32640156672 bytes. Q:54129-55185 |
| PLE head bias | **5120** if counted as an uncached read | `[16,160]` fp16; normally resident/cached, not sixteen row scales. Q:55155-55162 |
| PLE hash constants | **280** if counted as an uncached read | 16 int64 offsets, 16 int64 moduli, 3 int64 multipliers, CPU-resident constants. Q:55163-55184 |

The 1632-byte row payload already includes the sixteen fp16 row scales. Disk page reads can be much larger than 1632 bytes; OS/cache/direct-I/O policy is separate from logical row bytes. Excluding tiny constant rereads, trunk streamed weights plus gathers total **4615344324 bytes**. [T/families/qwen4_exp/cuda/exl3_pack.py:137-171; E/modules/ngram_embedding.py:142-162; G/ngram.cpp:153-192]

### 5.3 Incremental rows and expert overlap

One routed expert's three matrices total **1862412 bytes**. For a combined R-row launch that truly reads dense weights once and each distinct expert once, define U_i(R) as the union of selected experts in layer i:

```text
Bwindow(R) = 3721379812 + 1862412 * sum_{i=0..47} |U_i(R)|
```

The constant includes all non-routed weights and one full LM-head read. One row gives U_i=10. Under **zero overlap between rows**, each extra row adds **893957760 bytes** until the finite 512-expert union saturates. Under **complete expert overlap**, extra routed-weight bytes are zero. Independent uniformly distributed top-10 sets give expected union `512*(1-(502/512)^R)` per layer and expected incremental bytes for row R+1 of `893957760*(502/512)^R`. This is an analytical assumption, not a measured routing distribution. Embedding and PLE gathers still add 6752 bytes per extra row, plus its activation/cache work.

A kernel that rereads dense matrices separately per row pays the whole one-row weight table per row instead. Row-invariant arithmetic does not automatically provide weight reuse. LM head may only be evaluated on a subset of rows during prefill, but verification usually needs all logits. [active expert counts: Q:264-368 and C:111-114; row/expert grouping: T/families/qwen4_exp/cuda/forward.py:335-352; last-row head: T/families/qwen4_exp/cuda/forward.py:400-424]

### 5.4 One MTP draft step

MTP is absent from Q, so these are **header-derived**. Its attention/shared/routed matrices have the same geometry and bytes as one trunk QSA block. Each weight is counted once even though fc_hidden processes four stream rows.

| Family | Bytes/draft | Evidence |
| --- | ---: | --- |
| Embedding input norm `[2560]` bf16 | 5120 | I:3860484-3860496 |
| Hidden input norm `[10240]` fp16 | 20480 | I:3860471-3860483 |
| fc_embedding `[2560,2560]`, EXL3 **4-bit** | 3287044 | I:3860549-3860600 |
| fc_hidden `[2560,2560]`, EXL3 **4-bit** | 3287044 | I:3860497-3860548 |
| Two HC sites, down/up/inject/norm | 26419200 | I:3940657-3940711,3941024-3941078; same shapes as Q:30-64 |
| QSA q+gate / k / v / o | 19690500 / 825348 / 825348 / 9847812 | I:3940790-3941023; same widths as Q:163533-163672 |
| QSA main q/k norms | 1024 | I:3940842-3940854,3940907-3940919 |
| QSA indexer / indexer norms | 620804 / 512 | I:3940712-3940789 |
| Router / shared scalar gate | 2621440 / 5120 | I:3940629-3940656 |
| Ten routed experts, up/gate/down | 18624120 | I:3940317-3940472; 3-bit expert table |
| Shared expert up/gate/down | 3091212 | I:3940473-3940628 |
| MTP final HC down/up/norm, patch | 13127680 | I:3955787-3955832 |
| **MTP body streamed subtotal** | **102299808** | sum above |
| Shared full EXL3 LM head | **397813764** | Q:2598871-2598905 |
| **Full-vocabulary MTP streamed total** | **500113572** | **0.465767 GiB** |
| Shared embedding-row gather | **5120** | Q:15-27 |

MTP body format totals: EXL3 **60099232**, fp16 plain **42193920**, bf16 plain **6656**. Including full shared head, EXL3 subtotal is **457912996**. Full-head MTP weights plus embedding gather total **500118692 bytes**. There are no PLE gathers. A resident/shared LM head is not extra model storage, but is still a logical read when scored on each separate draft step. Restricted-vocabulary requantized draft-head bandwidth cannot be determined from this EXL3 manifest and must not be silently substituted for the full-head total.

### 5.5 State and context traffic, separate from weights

For baseline bf16 caches, the 36 fp32 DeltaNet states total **113246208 bytes**, read and rewritten per serial token if not retained on chip. Raw qkv history is **2211840 bytes** across 36 layers; PLE's nine-row conv history is **184320 bytes**. These are working-set sizes, not additional checkpoint weights. QSA K+V logical read per layer is `selected_tokens*2048` bytes, shared across the 24 query heads in the ideal reuse model; all 12 layers cost at most **50405376 bytes** at 2051 selected tokens. Indexer pooled-K scoring reads `12*floor((t+1)/4)*256` bytes; unlike sparse main KV reads, this grows with context. MTP adds one QSA layer's corresponding traffic. [shapes: HF:432-437,473-489,681-692,831-838; T/families/qwen4_exp/cuda/gdn.cu:70-75,146-151; T/families/qwen4_exp/cuda/attention.py:228-245]

## 6. Audit reproducibility and acceptance tests

The audit kept its scripts outside the repository; this document is its only durable output. No GPU execution was performed. The audit script parsed seven range-fetched headers, the n-gram header and MTP patch, asserted every manifest tensor shape/byte count matched, and summed families including unlisted tensors. The original n-gram header is 21336 bytes including its 8-byte length, so the 280-byte constants are at file byte range `26456..26735`. The immutable artifact URLs are formed as:

```text
https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/resolve/
69e33439ae950f17bcbe95c98f117d80f759ab6d/<filename>
```

To reproduce header-only inspection, request bytes 0..7, decode little-endian uint64 n, then request bytes 8..(7+n), decode JSON. Each tensor payload size is `data_offsets[1]-data_offsets[0]`; attach `8+n` to obtain absolute file offsets. Sum the four EXL3 component tensors, not just the trellis. Include `ngram_embedding.safetensors` and `mtp_hyper_connection_mixer_patch.safetensors`, which are not fully described by the main index/manifest. Use the table's explicit counts to distinguish selected weights from full resident model storage. [T/families/qwen4_exp/cuda/exl3_pack.py:38-71]

Before declaring a gfx1151 kernel implementation correct, test at least:

- Hash vectors around BOS/EOS, including a current EOS and history crossing one or two previous EOS tokens; compare all 16 row IDs against stored constants.
- HC read/write on four unequal streams; centered gamma near zero; grouped norm versus full-width MTP norm.
- GDN heads 0,1,2,3,47, causal conv first four tokens, fp32 recurrence against scalar formulas, every accepted prefix of a multi-row window.
- QSA positions around 2047..2055 tokens and block boundaries, zero/tied scores, incomplete tails, a low-score newest **completed** block, and accepted-prefix rollback that un-completes a pooled block.
- Top-10 router tie ordering, small logit margins, shared expert placement, per-operation bf16 rounding, and 1-row versus 2/4/8/16-row logits for identical prefixes.
- MTP input alignment `(stack[t],token[t+1])`, recursive draft stack, dedicated QSA state and loading of the out-of-index mixer patch.

These are recommended implementation acceptance tests, not tests run on Crucible. The pinned HF source's SHA-256 was `797a18fd6dd76c574d237a5643759acdeb1c4d0f1c2508693f8fdabce0a19057`, matching Gufo's pinned source identity. A dependency-free execution of HF's multiplier/prime functions reproduced all 35 stored PLE hash constants. Research confidence is high for architecture and serialized byte accounting. Cross-engine bitwise equivalence is explicitly unsupported; no end-to-end GPU inference was run in this audit.

## Top five implementation traps

1. **MTP hidden norm is one 10240-wide norm**, unlike per-stream HC norms. ExLlamaV3 marks its grouped MTP input norm as unverified, and it differs from four other implementations.
2. **Native GDN mapping is h//3 and A_log**, not Gufo's converted h%16 and negative-A storage. All decay/state/L2 work stays fp32; output gate is sigmoid.
3. **QSA top-k means 512 completed blocks plus at most three tail tokens**, not a guaranteed recent window or exactly 2048 tokens. Stable score ties and incomplete-block rollback matter.
4. **Centered gamma, bf16 boundaries and accumulation order are semantic numerics**, not optional fusion details. FP32 router selection is TensorFold's deliberate policy, not literal HF eager rounding.
5. **The manifest is incomplete and `.trellis` has two codecs.** Include omitted conv/A/dt/MTP weights, load the MTP patch, gather the 5-bit PLE rows, and account for 1.281 GB of HC weights plus the 397.8 MB head.
