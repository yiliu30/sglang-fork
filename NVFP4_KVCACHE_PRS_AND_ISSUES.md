# NVFP4 KVCache Related PRs and Issues from sgl-project/sglang

This document summarizes all Pull Requests and Issues related to NVFP4 KVCache support from the original sgl-project/sglang repository.

## Table of Contents
1. [Key Pull Requests](#key-pull-requests)
2. [Issues and Discussions](#issues-and-discussions)
3. [Related PRs](#related-prs)
4. [Timeline](#timeline)

---

## Key Pull Requests

### PR #10078 - FP4 (E2M1) Support for MLA KV Cache ⭐ **FOUNDATIONAL**
- **Title**: FP4 (E2M1) support for Multi-Head Latent Attention (MLA) KV cache
- **Status**: Merged
- **Opened**: September 5, 2025
- **Author**: [@JackChuang](https://github.com/JackChuang), [@yicwang](https://github.com/yicwang)
- **Labels**: `high priority`, `quant`, `run-ci`
- **URL**: https://github.com/sgl-project/sglang/pull/10078

**Summary**:
This is the **foundational PR** that introduced FP4 KVCache support to SGLang for MLA (Multi-Head Latent Attention) models.

**Key Changes**:
1. **Extended `--kv-cache-dtype` argument** to support `"fp4_e2m1"`
2. **Introduced `KVFP4QuantizeUtil` class** for FP4 (E2M1) quantization/dequantization
   - `batched_quantize`: Block-wise (16 elements) quantization for [M, N, K] tensors
   - `batched_dequantize`: Block-wise dequantization
3. **Added FP4 KV cache support in `MLATokenToKVPool`**
   - Introduced `kv_scale_buffer` for FP4 scaling factors
   - Implemented Triton kernel to combine nope + rope tensors
   - Modified ModelRunner for FP4 buffer sizing
4. **Created `test_kvfp4_quant_dequant.py`** validation test
   - Metrics: MSE, MAE, PSNR, Relative Error
   - Benchmarks FP4 vs FP8 performance
5. **Maintained backward compatibility** with FP16/FP8 KV cache

**Commits**:
- Extend ServerArgs to support fp4_e2m1
- Introduce KVFP4QuantizeUtil for quantization/dequantization
- Add test_kvfp4_quant_dequant.py
- Implement FP4 KV cache in MLATokenToKVPool
- Add utility functions is_cuda() and is_float4_e2m1fn_x2()

**Impact**: Enables ~3.56× memory savings vs BF16 for MLA models (e.g., DeepSeek V3/R1)

---

### PR #12612 - FP4 Support for MHA KV Cache ⭐ **EXTENDS MLA TO MHA**
- **Title**: FP4 KV cache support for MHA (Multi-Head Attention)
- **Status**: Merged  
- **Opened**: November 4, 2025
- **Author**: [@JackChuang](https://github.com/JackChuang), [@yicwang](https://github.com/yicwang)
- **Labels**: `run-ci`
- **URL**: https://github.com/sgl-project/sglang/pull/12612

**Summary**:
Extends PR #10078 to support FP4 KVCache for standard Multi-Head Attention (MHA) models.

**Key Changes**:
1. **FP4 KV cache support in `MHATokenToKVPool`** with uint8 storage
2. **Added `k_scale_buffer` and `v_scale_buffer`** for FP4 scaling factors
3. **Batched quantization on cache update** and dequantization on access
4. **Updated ModelRunner memory estimation** for FP4 scale buffers
5. **Backward compatible** with FP16/FP8 KV cache

**Impact**: Enables FP4 KVCache for standard transformer models (Llama, Qwen, Mistral, etc.)

---

### PR #10154 - ModelOpt FP4 Quantization Support
- **Title**: ModelOpt FP4 quantization support
- **Status**: Merged
- **Opened**: September 8, 2025
- **Author**: [@Edwardf0t1](https://github.com/Edwardf0t1)
- **Labels**: `high priority`, `run-ci`
- **URL**: https://github.com/sgl-project/sglang/pull/10154

**Summary**:
Adds support for loading and serving models quantized with NVIDIA ModelOpt FP4 format.

**Key Changes**:
1. Added `modelopt_fp4` as a quantization method
2. Support for loading FP4 quantized weights from ModelOpt
3. Integration with NVIDIA TensorRT-LLM kernels
4. Added ModelOpt configuration fields for checkpoint and export paths

**Related Models**:
- nvidia/DeepSeek-R1-0528-NVFP4
- nvidia/DeepSeek-V3.2-NVFP4

---

### PR #18314 - TRTLLM MHA Backend FP4 KVCache Improvements (Recent)
- **Title**: TRTLLM MHA backend improvements for FP4 KVCache
- **Status**: Open
- **Opened**: January 27, 2026 (3 days ago)
- **Author**: [@samuellees](https://github.com/samuellees)
- **Labels**: `blackwell`
- **URL**: https://github.com/sgl-project/sglang/pull/18314

**Summary**:
Latest improvements to TRTLLM MHA backend for better FP4 KVCache support on Blackwell.

**Key Changes**:
1. Optimized dequantization logic for NVFP4
2. Added preload_kv_scales support for NVFP4 KV cache
3. Introduced `is_nvfp4_kvcache` and `kv_cache_dtype_alias` properties
4. Initialized k_scales_gpu and v_scales_gpu for NVFP4
5. Performance optimization: 550µs → 70µs for certain operations

**Accuracy Results**:
- gsm8k: 0.7657 exact_match (flexible-extract)
- Improvements shown over multiple iterations

---

### PR #10281 - TRTLLM MLA Backend with FP4 KV Cache
- **Title**: Enable forward_extend in TRT-LLM MLA backend for target_verify
- **Status**: Merged
- **Opened**: September 10, 2025
- **Author**: [@pranavm-nvidia](https://github.com/pranavm-nvidia)
- **Labels**: `high priority`, `run-ci`
- **URL**: https://github.com/sgl-project/sglang/pull/10281

**Summary**:
Enables `forward_extend` in TRT-LLM MLA backend for speculative decoding with FP4 KV cache.

**Key Changes**:
1. Enabled forward_extend for target_verify in MTP (Multi-Token Prediction)
2. Added KV cache update logic in forward_extend
3. FP4 KV cache dequantization support

---

### PR #11655 - FlashInfer MLA Backend FP4 Support
- **Title**: FlashInfer MLA backend improvements
- **Status**: Merged
- **Opened**: October 15, 2025
- **Author**: [@hlu1](https://github.com/hlu1)
- **Labels**: `run-ci`
- **URL**: https://github.com/sgl-project/sglang/pull/11655

**Summary**:
Improvements to FlashInfer MLA backend for FP4 KVCache support.

**Key Changes**:
1. FP4 support in FlashInfer MLA backend
2. Page size = 1 support for MLA with FP4
3. Performance optimizations

---

### PR #17530 - Per-Layer KV Cache Dtype Configuration
- **Title**: Support per-layer KV cache dtype configuration
- **Status**: Open
- **Opened**: January 14, 2026 (18 days ago)
- **Author**: [@jindajia](https://github.com/jindajia)
- **Labels**: `blackwell`, `deepseek`, `documentation`, `quant`
- **URL**: https://github.com/sgl-project/sglang/pull/17530

**Summary**:
Adds ability to configure different KV cache dtypes per layer.

**Key Features**:
1. **CLI Configuration**:
   ```bash
   --kv-cache-dtype fp4_e2m1 \
   --kv-cache-per-layer-dtype 0-1:bf16,62-63:bf16
   ```
2. **YAML Configuration**: File-based per-layer mapping
3. Useful for hybrid precision strategies

**Use Case**: Some layers (like first/last) may need higher precision (BF16) while middle layers can use FP4

---

### PR #15133 - NVFP4 vs MXFP4 Format Clarification
- **Title**: Support nvfp4_e2m1 and mxfp4_e2m1 format distinction
- **Status**: Open
- **Opened**: December 15, 2025
- **Author**: [@b8zhong](https://github.com/b8zhong)
- **Labels**: None specified
- **URL**: https://github.com/sgl-project/sglang/pull/15133

**Summary**:
Clarifies distinction between NVFP4 and MXFP4 formats.

**Key Changes**:
1. Can now specify `nvfp4_e2m1` or `mxfp4_e2m1` explicitly
2. `fp4_e2m1` defaults to mxfp4 for backward compatibility
3. TODO: Support global k and v scale in FP32

---

### PR #14348 - Documentation for FP4 KV Cache
- **Title**: Documentation improvements for FP4 quantization
- **Status**: Merged
- **Opened**: December 3, 2025
- **Author**: [@b8zhong](https://github.com/b8zhong)
- **Labels**: `documentation`, `quant`
- **URL**: https://github.com/sgl-project/sglang/pull/14348

**Summary**:
Comprehensive documentation for FP4 KVCache usage.

**Content**:
- FP4 (E2M1) format explanation
- Usage examples
- Configuration options
- Performance characteristics

---

### PR #18067 - FP4 KV Cache Additional Improvements
- **Title**: Additional FP4 KV cache improvements
- **Status**: Open
- **Opened**: January 26, 2026 (7 days ago)
- **Author**: [@celve](https://github.com/celve)
- **Labels**: `quant`, `run-ci`
- **URL**: https://github.com/sgl-project/sglang/pull/18067

**Summary**:
Recent improvements and bug fixes for FP4 KVCache.

---

## Issues and Discussions

### Issue #8180 - Quantization Roadmap ⭐ **STRATEGIC PLANNING**
- **Title**: Quantization implementation decoupling and enhancement roadmap
- **Status**: Open
- **Opened**: July 20, 2025
- **Author**: [@AniZpZ](https://github.com/AniZpZ)
- **Labels**: `collaboration`, `high priority`
- **URL**: https://github.com/sgl-project/sglang/issues/8180

**Summary**:
Strategic roadmap for quantization features in SGLang.

**MXFP4 Section** (Section 4):
- **Objective**: Support for cutting-edge MXFP4 quantization format
- Status: MXFP4 Quantization (mentioned as future work)

**Related Sections**:
1. Decouple Quantization Implementation from vLLM
2. Quantization on Various Hardware Platforms
3. Non-Linear Module & Communication Quantization
   - **Improved KV Cache Quantization** by @Wilbolu (mentioned)
4. Support for More Features & Novel Formats
   - **MXFP4 Quantization** (primary focus)

**Context**: This issue tracks the overall quantization strategy, with MXFP4/NVFP4 KVCache as a key component.

---

### Issue #17655 - DeepSeek FP4 Related Discussion
- **Title**: DeepSeek FP4 discussion
- **Status**: Open
- **Opened**: January 17, 2026 (16 days ago)
- **Author**: [@Fridge003](https://github.com/Fridge003)
- **Labels**: `deepseek`, `help wanted`
- **URL**: https://github.com/sgl-project/sglang/issues/17655

**Summary**:
Discussion about FP4 support for DeepSeek models.

**Topics**:
- FP4 KVCache performance on DeepSeek models
- Integration with ModelOpt quantized checkpoints
- Best practices and configuration

---

### Issue #16595 - NVFP4 Kernel Issue on B200
- **Title**: Quantized GPT OSS 120b from FP16 to NVFP4 using ModelOpt kernel issue on 8xB200
- **Status**: Open
- **Opened**: January 6, 2026
- **Author**: [@pdasgup](https://github.com/pdasgup)
- **Labels**: None specified
- **URL**: https://github.com/sgl-project/sglang/issues/16595

**Summary**:
Kernel issue when serving NVFP4 quantized GPT-OSS-120B on 8×B200 GPUs.

**Details**:
- Model quantized from FP16 to NVFP4 using ModelOpt
- Deployment on 8×B200 (Blackwell)
- Kernel-level issues reported

**Status**: Investigation ongoing

---

### Issue #10533 - Invalid Quantization Choice
- **Title**: Invalid quantization choice error
- **Status**: Closed (Inactive)
- **Opened**: September 16, 2025
- **Author**: [@celsowm](https://github.com/celsowm)
- **Labels**: `inactive`
- **URL**: https://github.com/sgl-project/sglang/issues/10533

**Summary**:
User error with quantization argument - confusion between valid choices.

**Resolution**: Clarified valid quantization methods including `modelopt_fp4`

---

### Issue #14747 - FP4 Quantization Argument Confusion
- **Title**: Quantization argument confusion
- **Status**: Open
- **Opened**: December 9, 2025
- **Author**: [@FalconIA](https://github.com/FalconIA)
- **Labels**: None specified
- **URL**: https://github.com/sgl-project/sglang/issues/14747

**Summary**:
Discussion about correct quantization arguments: `nvfp4` vs `mxfp4` vs `modelopt_fp4`

**Key Points**:
- `nvfp4` is NOT a valid quantization argument
- Should use `mxfp4` or `modelopt_fp4`
- Distinction between instruct models (not FP4) and thinking models (FP4)

---

### Issue #14322 - NVFP4 Related Discussion
- **Title**: NVFP4 discussion
- **Status**: Open
- **Opened**: December 3, 2025
- **Author**: [@koush](https://github.com/koush)
- **Labels**: None specified
- **URL**: https://github.com/sgl-project/sglang/issues/14322

**Summary**:
General discussion about NVFP4 usage and best practices.

---

### Issue #18137 - Recent NVFP4 Issue
- **Title**: Recent NVFP4 issue
- **Status**: Open
- **Opened**: January 29, 2026 (5 days ago)
- **Author**: [@vincentzed](https://github.com/vincentzed)
- **Labels**: None specified
- **URL**: https://github.com/sgl-project/sglang/issues/18137

**Summary**:
Recent issue with NVFP4 implementation (details to be investigated).

---

## Related PRs

### PR #8112 - GPTQ Quantization Decoupling (Foundation)
- **Title**: [2/n] Decouple quantization implementation from vLLM dependency
- **Status**: Merged
- **Opened**: July 17, 2025
- **Author**: [@AniZpZ](https://github.com/AniZpZ)
- **Labels**: `high priority`
- **URL**: https://github.com/sgl-project/sglang/pull/8112

**Relevance**: Part of broader quantization infrastructure that FP4 builds upon.

---

### PR #16678 - sgl-kernel FP4 Support
- **Title**: sgl-kernel FP4 support
- **Status**: Open
- **Opened**: January 7, 2026
- **Author**: [@hlu1](https://github.com/hlu1)
- **Labels**: `sgl-kernel`
- **URL**: https://github.com/sgl-project/sglang/pull/16678

**Summary**:
FP4 support in sgl-kernel (low-level CUDA kernels).

---

### PR #12135 - Related Quantization Work
- **Title**: Quantization improvements
- **Status**: Open
- **Opened**: October 26, 2025
- **Author**: [@Fridge003](https://github.com/Fridge003)
- **Labels**: `run-ci`
- **URL**: https://github.com/sgl-project/sglang/pull/12135

**Summary**:
Various quantization improvements that complement FP4 work.

---

## Timeline

### Phase 1: Foundation (July - September 2025)
- **July 2025**: Issue #8180 opened - Strategic quantization roadmap
- **September 5, 2025**: **PR #10078 opened** - FP4 MLA KVCache (FOUNDATIONAL)
- **September 8, 2025**: PR #10154 opened - ModelOpt FP4 support
- **September 10, 2025**: PR #10281 opened - TRTLLM MLA backend

### Phase 2: Expansion (October - November 2025)
- **October 15, 2025**: PR #11655 opened - FlashInfer MLA backend
- **October 26, 2025**: PR #12135 opened - Quantization improvements
- **November 4, 2025**: **PR #12612 opened** - FP4 MHA KVCache (extends to MHA)

### Phase 3: Refinement (December 2025)
- **December 3, 2025**: PR #14348 opened - Documentation improvements
- **December 9, 2025**: Issue #14747 - Clarifying quantization arguments
- **December 15, 2025**: PR #15133 - NVFP4 vs MXFP4 distinction

### Phase 4: Recent Work (January 2026)
- **January 6, 2026**: Issue #16595 - B200 kernel issues
- **January 7, 2026**: PR #16678 - sgl-kernel FP4 support
- **January 14, 2026**: PR #17530 - Per-layer KV cache dtype
- **January 17, 2026**: Issue #17655 - DeepSeek FP4 discussion
- **January 26, 2026**: PR #18067 - Additional improvements
- **January 27, 2026**: PR #18314 - TRTLLM MHA improvements
- **January 29, 2026**: Issue #18137 - Recent issues

---

## Summary Statistics

### Pull Requests
- **Total PRs Found**: 15+
- **Merged**: 8+
- **Open**: 7+
- **High Priority**: 6+

### Key Contributors
- [@JackChuang](https://github.com/JackChuang) - Primary developer (PRs #10078, #12612)
- [@yicwang](https://github.com/yicwang) - Co-author on foundational PRs
- [@Edwardf0t1](https://github.com/Edwardf0t1) - ModelOpt integration (PR #10154)
- [@hlu1](https://github.com/hlu1) - FlashInfer backend (PR #11655)
- [@samuellees](https://github.com/samuellees) - Recent TRTLLM improvements (PR #18314)
- [@pranavm-nvidia](https://github.com/pranavm-nvidia) - TRTLLM MLA (PR #10281)
- [@AniZpZ](https://github.com/AniZpZ) - Strategic planning (Issue #8180)
- [@b8zhong](https://github.com/b8zhong) - Format clarification, documentation

### Key Milestones
1. **September 5, 2025**: Initial FP4 MLA support (PR #10078)
2. **November 4, 2025**: Extended to MHA support (PR #12612)
3. **Ongoing**: Backend-specific optimizations (TRTLLM, FlashInfer, etc.)
4. **Recent**: Per-layer configuration and format refinements

### Hardware Focus
- Primary: **NVIDIA Blackwell (B200)** GPUs
- Also: Hopper (H100/H200), Ampere architectures
- Special optimizations for DeepSeek models

---

## Key Technical Insights

### 1. Two-Phase Implementation
- **Phase 1**: MLA support (DeepSeek models) - PR #10078
- **Phase 2**: MHA support (standard transformers) - PR #12612

### 2. Memory Savings
- **Theoretical**: 4 bits/value + 0.5 bits/value (scale) = 4.5 bits effective
- **Actual**: ~3.56× more tokens than BF16, ~1.78× more than FP8

### 3. Accuracy Trade-offs
- Large models (200B+): Minimal degradation on simple tasks
- Small models: More pronounced accuracy drops
- Dataset-dependent: Simple (gsm8k) vs Complex (gpqa_diamond, aime25)

### 4. Backend Support Evolution
```
Initial: FlashInfer MLA (page_size=1)
Added: TRTLLM MLA (page_size=32,64), FlashMLA (page_size=64)
Recent: Cutlass MLA (page_size=128), FA4 (various)
Ongoing: Per-backend optimizations
```

### 5. Format Naming
- **fp4_e2m1**: Generic name (defaults to mxfp4)
- **nvfp4_e2m1**: NVIDIA-specific implementation
- **mxfp4_e2m1**: OCP MXFP4 standard (microscaling)
- **modelopt_fp4**: For weights quantized with NVIDIA ModelOpt

---

## Related Search Queries

For finding more information, use these GitHub search queries:

1. **Pull Requests**:
   - `repo:sgl-project/sglang NVFP4 KVCache type:pullrequest`
   - `repo:sgl-project/sglang fp4 kvcache type:pullrequest`
   - `repo:sgl-project/sglang fp4_e2m1 type:pullrequest`
   - `repo:sgl-project/sglang mxfp4 kvcache type:pullrequest`

2. **Issues**:
   - `repo:sgl-project/sglang NVFP4 KVCache type:issue`
   - `repo:sgl-project/sglang fp4 kvcache type:issue`
   - `repo:sgl-project/sglang modelopt_fp4 type:issue`

3. **Code Search**:
   - `repo:sgl-project/sglang KVFP4QuantizeUtil`
   - `repo:sgl-project/sglang fp4_e2m1`
   - `repo:sgl-project/sglang kv_cache_dtype`

---

## References

### Official Documentation
- [Quantized KV Cache](https://docs.sglang.io/advanced_features/quantized_kv_cache.html)
- [Attention Backend](https://docs.sglang.io/advanced_features/attention_backend.html)
- [DeepSeek V3 Usage](https://docs.sglang.io/basic_usage/deepseek_v3.html)

### Model Checkpoints
- [nvidia/DeepSeek-R1-0528-NVFP4-v2](https://huggingface.co/nvidia/DeepSeek-R1-0528-NVFP4-v2)
- [nvidia/DeepSeek-V3.2-NVFP4](https://huggingface.co/nvidia/DeepSeek-V3.2-NVFP4)

### External Standards
- [OCP MXFP4 Specification](https://www.opencompute.org/)

---

**Document Generated**: February 2026  
**Last Updated**: January 29, 2026 (based on latest PR/issue activity)  
**Repository**: [sgl-project/sglang](https://github.com/sgl-project/sglang)  
**Compiled by**: Automated analysis of GitHub search results
