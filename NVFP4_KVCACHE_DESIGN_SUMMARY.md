# NVFP4 KVCache Support in SGLang - Design and Implementation Summary

## Executive Summary

NVFP4 (NVIDIA FP4) KVCache support in SGLang provides a memory optimization technique that enables approximately **3.56× more tokens** to be cached compared to BF16, using 4-bit floating-point quantization with block-based microscaling (MXFP4 format). This implementation is particularly optimized for NVIDIA Blackwell architecture (B200 GPUs) and supports both Multi-Head Attention (MHA) and Multi-Head Latent Attention (MLA) models.

## 1. Overview

### 1.1 What is NVFP4 KVCache?

NVFP4 KVCache is a quantized key-value cache storage format that uses the OCP (Open Compute Project) MXFP4 (Microscaling FP4) standard:
- **Format**: E2M1 (1 sign bit, 2 exponent bits, 1 mantissa bit)
- **Block Size**: 16 elements per block (SGLang's implementation)
- **Scaling**: Each block of 16 elements shares a single 8-bit exponential scaling factor
- **Memory Savings**: ~3.56× more tokens than BF16, ~1.78× more than FP8

### 1.2 Key Benefits

1. **Memory Efficiency**: Significantly reduces KV cache memory footprint
2. **Throughput**: Enables longer context lengths or more concurrent requests
3. **Dynamic Scaling**: Scaling factors computed automatically on-the-fly (no pre-quantization needed)
4. **Hardware Optimized**: Specifically tuned for NVIDIA Blackwell (B200) GPUs

## 2. Technical Design

### 2.1 Quantization Format (E2M1)

The E2M1 format represents values using:
```
E2M1_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6]  # 8 possible values
E2M1_BOUNDS = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5]  # Decision boundaries
E2M1_MAX = 6.0  # Maximum representable value
```

Each value requires 4 bits:
- 1 bit for sign
- 3 bits for magnitude (8 values: 0-7)

### 2.2 Block-Based Microscaling

**Key Characteristics:**
- Tensors divided into blocks of 16 consecutive elements
- Each block shares one 8-bit exponential scaling factor
- Scale factor: `scale_exp = ceil(log2(max(block) / 6.0))`
- Stored as: `scale_factor_uint8 = scale_exp + 127`

**Memory Layout:**
```
Original Tensor:  [B, M, N]  (BF16)
                     ↓
Quantized Tensor: [B, M, N/2]  (uint8, packed)
Scale Factors:    [B, M*N/16]  (uint8)
```

### 2.3 Packing Strategy

Two FP4 values are packed into one uint8:
```python
# Packing
packed = (high_nibble << 4) | low_nibble

# Unpacking
low_nibble = packed & 0x0F
high_nibble = (packed >> 4) & 0x0F
```

## 3. Implementation Architecture

### 3.1 Core Components

#### 3.1.1 KVFP4QuantizeUtil Class
Location: `python/sglang/srt/layers/quantization/kvfp4_tensor.py`

**Key Methods:**
```python
class KVFP4QuantizeUtil:
    @staticmethod
    @torch.compile
    def batched_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to KVFP4 format
        Args:
            tensor: [B, M, N] input tensor
        Returns:
            quant_tensor: [B, M, N/2] quantized tensor
            scale_factors: [B, M*N/16] scale factors
        """
        # 1. Reshape to [B, M*N/16, 16] for block-wise quantization
        # 2. Compute per-block scale factors
        # 3. Apply scaling
        # 4. Quantize to FP4 (E2M1)
        # 5. Pack two FP4 values into one uint8
        
    @staticmethod
    @torch.compile
    def batched_dequantize(
        quant_tensor: torch.Tensor,
        scale_factors: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize KVFP4 tensor
        Args:
            quant_tensor: [B, M, N/2] quantized tensor
            scale_factors: [B, M*N/16] scale factors
        Returns:
            Dequantized tensor: [B, M, N]
        """
        # 1. Unpack uint8 to two FP4 values
        # 2. Extract sign and magnitude
        # 3. Convert to float values using E2M1_VALUES lookup
        # 4. Reshape for block-wise scaling
        # 5. Apply scale factors
```

**Performance Optimizations:**
- Uses `@torch.compile` for JIT compilation
- Pure tensor operations (CUDA Graph safe)
- Efficient bit operations for packing/unpacking
- Vectorized lookups using tensor indexing

#### 3.1.2 Memory Pool Integration
Location: `python/sglang/srt/mem_cache/memory_pool.py`

**Integration Points:**

1. **Buffer Storage:**
```python
# Separate buffers for quantized data and scale factors
self.k_buffer[layer_id]        # Quantized K cache
self.k_scale_buffer[layer_id]  # K scale factors
self.v_buffer[layer_id]        # Quantized V cache
self.v_scale_buffer[layer_id]  # V scale factors
```

2. **Write Path (set_kv_buffer):**
```python
def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None):
    # 1. Apply FP8 scale factors if provided
    if k_scale is not None:
        cache_k.div_(k_scale)
    
    # 2. Quantize to FP4
    cache_k_fp4, cache_k_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k)
    cache_v_fp4, cache_v_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_v)
    
    # 3. Store in buffers
    self.k_buffer[layer_id][loc] = cache_k_fp4.view(self.store_dtype)
    self.k_scale_buffer[layer_id][loc] = cache_k_fp4_sf.view(self.store_dtype)
```

3. **Read Path (_get_key_buffer):**
```python
def _get_key_buffer(self, layer_id):
    if self.store_dtype != self.dtype:
        cache_k_fp4 = self.k_buffer[layer_id].view(torch.uint8)
        cache_k_fp4_sf = self.k_scale_buffer[layer_id]
        
        # Dequantize on-the-fly
        cache_k_dequant = KVFP4QuantizeUtil.batched_dequantize(
            cache_k_fp4, cache_k_fp4_sf
        )
        return cache_k_dequant
```

#### 3.1.3 MLA-Specific Support

For Multi-Head Latent Attention (DeepSeek models):
```python
def set_mla_kv_buffer(self, layer, loc, cache_k_nope, cache_k_rope):
    # Separate quantization for nope and rope components
    cache_k_nope_fp4, cache_k_nope_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k_nope)
    cache_k_rope_fp4, cache_k_rope_fp4_sf = KVFP4QuantizeUtil.batched_quantize(cache_k_rope)
    
    # Store both components with their scale factors
```

### 3.2 Attention Backend Support

#### Support Matrix

**MHA Backends (Standard Attention):**
- ✅ **FA4 (FlashAttention 4)**: Full support with page_size=128
- ✅ **Triton**: Full support
- ✅ **Torch Native (SDPA)**: Full support
- ✅ **FlexAttention**: Full support
- ✅ **TRTLLM MHA**: Full support with page_size ∈ {16, 32, 64}
- ❌ **FlashInfer**: Not supported
- ❌ **FA3**: Not supported

**MLA Backends (Multi-Head Latent Attention):**
- ✅ **FlashInfer MLA**: Full support with page_size=1
- ✅ **FlashMLA**: Full support with page_size=64
- ✅ **Cutlass MLA**: Full support with page_size=128
- ✅ **TRTLLM MLA (Blackwell)**: Full support with page_size ∈ {32, 64}
- ✅ **FA4**: Full support with page_size=1
- ❌ **FA3**: Not supported
- ❌ **Triton**: Not supported

#### Hybrid Attention Support

SGLang supports mixing different backends for prefill and decode:
```bash
# Example: FA4 for prefill, TRTLLM MLA for decode
python3 -m sglang.launch_server \
  --model-path nvidia/DeepSeek-R1-FP4 \
  --tp 8 \
  --attention-backend trtllm_mla \
  --prefill-attention-backend fa4 \
  --kv-cache-dtype fp4_e2m1
```

### 3.3 CUDA Graph Compatibility

The implementation is fully CUDA Graph compatible:
- All tensor operations are pure (no Python control flow)
- Uses `@torch.compile` for kernel fusion
- Overlapped memory copies with alternative streams:
```python
if get_is_capture_mode() and self.alt_stream is not None:
    current_stream = self.device_module.current_stream()
    self.alt_stream.wait_stream(current_stream)
    # Perform quantization and copy in alt_stream
```

## 4. Usage Guide

### 4.1 Server Arguments

**Basic Usage:**
```bash
# Enable FP4 KV cache
python3 -m sglang.launch_server \
    --model-path nvidia/DeepSeek-R1-0528-NVFP4 \
    --kv-cache-dtype fp4_e2m1
```

**With Quantized Weights (NVFP4 Checkpoint):**
```bash
python -m sglang.launch_server \
    --model nvidia/DeepSeek-V3.2-NVFP4 \
    --tp 4 \
    --quantization modelopt_fp4 \
    --moe-runner-backend flashinfer_trtllm \
    --kv-cache-dtype fp4_e2m1
```

**Advanced Configuration:**
```bash
python3 -m sglang.launch_server \
    --model-path nvidia/DeepSeek-R1-0528-NVFP4 \
    --tp 8 \
    --attention-backend trtllm_mla \
    --kv-cache-dtype fp4_e2m1 \
    --quantization modelopt_fp4 \
    --moe-runner-backend flashinfer_trtllm
```

### 4.2 Hardware Requirements

**Minimum Requirements:**
- CUDA 12.8+ (for NVFP4 support)
- PyTorch 2.8.0+
- NVIDIA Blackwell GPUs (B200) recommended

**Supported Configurations:**
| Model | Weight Type | Configuration |
|-------|-------------|---------------|
| DeepSeek-R1-0528 | NVFP4 | 8 × B200 or 4 × B200 |
| DeepSeek-V3.2 | NVFP4 | 8 × B200 or 4 × B200 |

### 4.3 No Pre-Quantization Required

Unlike FP8 KVCache, FP4 requires no external scaling factors:
- **FP8**: Requires `k_scale` and `v_scale` from checkpoint or JSON file
- **FP4**: Scaling factors computed automatically during quantization

```python
# FP8 requires pre-computed scales
--quantization-param-path kv_scales.json

# FP4 works out of the box (no external files needed)
--kv-cache-dtype fp4_e2m1
```

## 5. Performance Characteristics

### 5.1 Memory Savings

**Theoretical Savings:**
```
BF16:  16 bits/value
FP8:    8 bits/value  (~2.0× savings)
FP4:    4 bits/value + 0.5 bits/value (scale overhead)
       = 4.5 bits/value (~3.56× savings vs BF16, ~1.78× vs FP8)
```

**Actual Memory Layout:**
```
For N elements:
- BF16:  N × 2 bytes
- FP8:   N × 1 byte + scales
- FP4:   N × 0.5 bytes + N/16 bytes (scales)
```

### 5.2 Accuracy Impact

Based on preliminary accuracy tests from PRs #10078 (MLA) and #12612 (MHA):

**Large Models (200B+ parameters):**
- Simple datasets (gsm8k): Minimal degradation (~0.3% drop)
- Complex datasets (gpqa_diamond): 2-4% accuracy drop
- Challenging datasets (aime25): 10-25% accuracy drop

**Example Results (Qwen3-235B-A22B):**
| Dataset | KV16 | KV8 (FP8) | KV4 (FP4) |
|---------|------|-----------|-----------|
| gsm8k | 0.9168 | 0.9181 | 0.9186 |
| gpqa_diamond | 0.7010 | 0.6899 | 0.6778 |
| aime25 | 0.7733 | 0.7333 | 0.6000 |

**Smaller Models (GPT-OSS-120B):**
- Simple datasets: ~0.1% drop
- Complex datasets: Up to 15-20% accuracy drop

**Key Observations:**
1. Model size matters: Large models tolerate FP4 better
2. Dataset complexity matters: Simple tasks show minimal degradation
3. Context length matters: Longer contexts may show more degradation

### 5.3 Performance Considerations

**Pros:**
- 3.56× memory savings enable much longer contexts
- Dynamic scaling eliminates offline quantization step
- CUDA Graph compatible for low latency
- Fused quantization/dequantization with attention kernels

**Cons:**
- Accuracy degradation on complex tasks
- Requires backend support (not all attention backends supported)
- CPU overhead for quantization/dequantization if not fused
- Works best on Blackwell architecture

### 5.4 Benchmark Performance

**Test Configuration:**
Tensor shapes: `[M, N, K]` where:
- MLA: `[M, 1, 576]` (DeepSeek models)
- MHA: `[M, 8, 64]` (GPT-OSS models)

**Latency Measurements:**
```python
# Quantization time: ~similar to FP8 cast
# Dequantization time: ~2-3× slower than FP8 cast
# Overall: Acceptable for throughput-oriented workloads
```

**Metrics Tracked:**
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Peak Signal-to-Noise Ratio (PSNR)
- Relative Error

Test file: `python/sglang/test/test_kvfp4_quant_dequant.py`

## 6. Implementation Details

### 6.1 Quantization Algorithm

```python
def batched_quantize(tensor):
    # Step 1: Reshape to blocks
    B, M, N = tensor.shape
    reshaped = tensor.view(B, M*N//16, 16)  # [B, num_blocks, 16]
    
    # Step 2: Compute per-block scale factors
    block_max = reshaped.abs().max(dim=-1, keepdim=True).values
    scale_exp = torch.ceil(torch.log2(torch.clamp(block_max / 6.0, min=1e-10)))
    scale_factors = (scale_exp + 127).to(torch.uint8)  # Bias by 127
    
    # Step 3: Scale values
    scaled = reshaped / torch.exp2(scale_exp)
    
    # Step 4: Quantize to E2M1
    sign_bits = (scaled < 0).to(torch.uint8) << 3
    abs_vals = scaled.abs()
    magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= E2M1_BOUNDS, dim=-1)
    fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)
    
    # Step 5: Pack two FP4 values into one uint8
    fp4_reshaped = fp4_vals.view(B, M, N)
    packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]
    
    return packed, scale_factors
```

### 6.2 Dequantization Algorithm

```python
def batched_dequantize(quant_tensor, scale_factors, dtype):
    B, M, N_half = quant_tensor.shape
    N = N_half * 2
    
    # Step 1: Unpack uint8 to two FP4 values
    fp4_vals = torch.empty(B, M, N, dtype=torch.uint8, device=quant_tensor.device)
    fp4_vals[..., 0::2] = quant_tensor & 0x0F
    fp4_vals[..., 1::2] = (quant_tensor >> 4) & 0x0F
    
    # Step 2: Extract sign and magnitude
    sign_mask = (fp4_vals & 0x08) != 0
    magnitude_idx = fp4_vals & 0x07
    
    # Step 3: Convert to float values
    float_vals = E2M1_VALUES[magnitude_idx.long()]
    float_vals = torch.where(sign_mask, -float_vals, float_vals)
    
    # Step 4: Reshape for block-wise scaling
    reshaped = float_vals.view(B, M*N//16, 16)
    
    # Step 5: Apply scale factors
    scale_exp = scale_factors.float() - 127  # Remove bias
    scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))
    
    return scaled.view(B, M, N).to(dtype)
```

### 6.3 Integration with Attention Layers

The FP4 quantization is transparent to attention layers:

```python
# In attention forward pass:
# 1. New KV computed in high precision (BF16)
# 2. Automatically quantized when written to cache
# 3. Automatically dequantized when read from cache
# 4. Attention computation happens in high precision

# Example flow:
new_k, new_v = compute_kv(hidden_states)  # BF16
memory_pool.set_kv_buffer(layer, loc, new_k, new_v)  # Quantizes to FP4
cached_k = memory_pool.get_key_buffer(layer_id)  # Dequantizes to BF16
output = attention_kernel(q, cached_k, cached_v)  # Compute in BF16
```

## 7. Testing and Validation

### 7.1 Unit Tests

**Quantization/Dequantization Tests:**
```bash
pytest python/sglang/test/test_kvfp4_quant_dequant.py
```

Tests cover:
- Accuracy metrics (MSE, MAE, PSNR, relative error)
- Various tensor shapes (MLA and MHA configurations)
- Performance benchmarking (vs FP8)

**Kernel Tests:**
```bash
pytest sgl-kernel/tests/test_fp4_quantize.py
pytest sgl-kernel/tests/test_fp4_gemm.py
```

**Integration Tests:**
```bash
pytest test/registered/quant/test_nvfp4_gemm.py
pytest test/registered/quant/test_deepseek_v3_fp4_4gpu.py
pytest test/registered/quant/test_deepseek_v32_fp4_4gpu.py
```

### 7.2 End-to-End Model Tests

**DeepSeek Models:**
- `test/registered/backends/test_deepseek_v3_fp4_cutlass_moe.py`
- `test/registered/backends/test_qwen3_fp4_trtllm_gen_moe.py`
- `test/registered/spec/eagle/test_deepseek_v3_fp4_mtp_small.py`

**Performance Tests:**
- `test/registered/perf/test_dpsk_r1_fp4_4gpu_perf.py`

### 7.3 Accuracy Validation

Accuracy tests on standard benchmarks:
```bash
# GSM8K
python3 benchmark/gsm8k/bench_sglang.py --num-shots 8 --parallel 1319

# Expected results with FP4:
# - Large models: >91% accuracy (minimal degradation)
# - Small models: May vary significantly
```

## 8. Best Practices

### 8.1 When to Use FP4 KVCache

**Recommended:**
- ✅ Large models (200B+ parameters)
- ✅ Simple to moderate reasoning tasks
- ✅ Throughput-oriented workloads
- ✅ Memory-constrained environments
- ✅ Blackwell (B200) GPUs

**Not Recommended:**
- ❌ Small models (<100B parameters)
- ❌ Complex reasoning tasks requiring high accuracy
- ❌ Latency-critical applications (without fused kernels)
- ❌ Pre-Blackwell architectures (Hopper, Ampere)

### 8.2 Configuration Checklist

```bash
# 1. Check CUDA version
nvcc --version  # Should be 12.8+

# 2. Check PyTorch version
python -c "import torch; print(torch.__version__)"  # Should be 2.8.0+

# 3. Verify backend support
--attention-backend {trtllm_mla|flashinfer_mla|flashmla|cutlass_mla|fa4}

# 4. Enable FP4 KV cache
--kv-cache-dtype fp4_e2m1

# 5. For quantized weights, specify quantization
--quantization modelopt_fp4

# 6. For MoE models, specify MoE runner
--moe-runner-backend flashinfer_trtllm
```

### 8.3 Accuracy Evaluation

Always evaluate on your specific workload:
```bash
# Run accuracy tests
python benchmark/gsm8k/bench_sglang.py --num-shots 8

# Compare with baseline
# FP16/BF16 baseline
python -m sglang.launch_server --model MODEL --kv-cache-dtype auto

# FP8 baseline  
python -m sglang.launch_server --model MODEL --kv-cache-dtype fp8_e4m3

# FP4 test
python -m sglang.launch_server --model MODEL --kv-cache-dtype fp4_e2m1
```

### 8.4 Debugging Tips

**Common Issues:**

1. **Backend Not Supporting FP4:**
```
Error: Backend 'flashinfer' does not support fp4_e2m1 kv_cache_dtype
Solution: Use supported backend (trtllm_mla, flashinfer_mla, fa4, etc.)
```

2. **CUDA Version Too Old:**
```
Error: NVFP4 requires CUDA 12.8+
Solution: Upgrade CUDA toolkit
```

3. **Performance Degradation:**
```
Issue: Slow inference despite memory savings
Cause: Quantization not fused with attention
Solution: Use backend with fused support (trtllm_mla recommended)
```

## 9. Future Work and Limitations

### 9.1 Current Limitations

1. **Backend Support**: Not all attention backends support FP4
2. **Accuracy**: May degrade on complex reasoning tasks
3. **Architecture**: Optimized primarily for Blackwell (B200)
4. **Block Size**: Fixed at 16 elements (no tuning option)

### 9.2 Potential Improvements

1. **Adaptive Block Sizing**: Dynamically adjust block size based on data distribution
2. **Mixed Precision**: Use FP4 for some layers, FP8/BF16 for others
3. **Improved Quantization**: Better rounding strategies
4. **Broader Backend Support**: Add FP4 support to more backends
5. **CPU Fallback**: Efficient CPU implementation for non-Blackwell GPUs

### 9.3 Related Work

- **FP8 KVCache**: More mature, better accuracy, but 2× less memory efficient
- **INT4 Quantization**: Similar memory savings but requires different kernels
- **Sparse KVCache**: Complementary optimization focusing on attention patterns

## 10. References

### 10.1 Documentation

- [Quantized KV Cache Guide](docs/advanced_features/quantized_kv_cache.md)
- [Attention Backend Guide](docs/advanced_features/attention_backend.md)
- [DeepSeek V3 Usage](docs/basic_usage/deepseek_v3.md)
- [DeepSeek V3.2 Usage](docs/basic_usage/deepseek_v32.md)

### 10.2 Pull Requests

- PR #10078: MLA FP4 KVCache support
- PR #12612: MHA FP4 KVCache support

### 10.3 External Resources

- [OCP MXFP4 Specification](https://www.opencompute.org)
- [DeepSeek MLA Paper](https://arxiv.org/pdf/2405.04434)
- [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)

### 10.4 Model Checkpoints

- [NVIDIA DeepSeek-R1-0528-NVFP4-v2](https://huggingface.co/nvidia/DeepSeek-R1-0528-NVFP4-v2)
- [NVIDIA DeepSeek-V3.2-NVFP4](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4)

## 11. Conclusion

NVFP4 KVCache support in SGLang provides a powerful memory optimization technique that enables significantly longer context lengths and higher throughput for large language model inference. While it introduces some accuracy trade-offs, it is particularly effective for large models (200B+ parameters) on simpler tasks and is well-suited for throughput-oriented workloads on NVIDIA Blackwell GPUs.

The implementation leverages block-based microscaling with dynamic scale factor computation, eliminating the need for offline quantization while maintaining CUDA Graph compatibility. Integration with SGLang's memory pool is seamless, and support for both MHA and MLA attention patterns makes it versatile across different model architectures.

For production deployments, careful evaluation of the accuracy/memory trade-off is recommended, with FP8 being a safer choice for accuracy-critical applications and FP4 excelling in memory-constrained scenarios where moderate accuracy degradation is acceptable.
