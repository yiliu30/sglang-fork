# NVFP4 KVCache Design Summary

This document provides a comprehensive design and implementation summary for NVFP4 (NVIDIA FP4) KVCache support in SGLang.

## Document Location

**Main Summary:** [`NVFP4_KVCACHE_DESIGN_SUMMARY.md`](NVFP4_KVCACHE_DESIGN_SUMMARY.md)

## What's Covered

The summary document includes:

1. **Executive Overview** - High-level description of NVFP4 KVCache and its benefits
2. **Technical Design** - Detailed explanation of the E2M1 quantization format and block-based microscaling
3. **Implementation Architecture** - Core components, integration points, and code examples
4. **Attention Backend Support** - Compatibility matrix for MHA and MLA backends
5. **Usage Guide** - Server arguments, hardware requirements, and configuration examples
6. **Performance Characteristics** - Memory savings, accuracy impact, and benchmarks
7. **Testing & Validation** - Unit tests, integration tests, and accuracy validation
8. **Best Practices** - When to use FP4, configuration checklist, and debugging tips
9. **Future Work** - Current limitations and potential improvements
10. **References** - Links to documentation, PRs, and external resources

## Quick Links

### Implementation Files
- **Core Quantization Utility:** `python/sglang/srt/layers/quantization/kvfp4_tensor.py`
- **Memory Pool Integration:** `python/sglang/srt/mem_cache/memory_pool.py`
- **Tests:** `python/sglang/test/test_kvfp4_quant_dequant.py`

### Documentation
- **Quantized KV Cache Guide:** `docs/advanced_features/quantized_kv_cache.md`
- **Attention Backend Guide:** `docs/advanced_features/attention_backend.md`
- **DeepSeek V3 Usage:** `docs/basic_usage/deepseek_v3.md`
- **DeepSeek V3.2 Usage:** `docs/basic_usage/deepseek_v32.md`

## Key Takeaways

- **Memory Savings:** ~3.56× more tokens than BF16, ~1.78× more than FP8
- **Hardware:** Optimized for NVIDIA Blackwell (B200) GPUs
- **Requirements:** CUDA 12.8+, PyTorch 2.8.0+
- **Use Cases:** Best for large models (200B+) on throughput-oriented workloads
- **Accuracy:** Minimal degradation on simple tasks, more pronounced on complex reasoning

## Example Usage

```bash
# Basic FP4 KV cache
python3 -m sglang.launch_server \
    --model-path nvidia/DeepSeek-R1-0528-NVFP4 \
    --kv-cache-dtype fp4_e2m1

# With quantized weights on Blackwell
python -m sglang.launch_server \
    --model nvidia/DeepSeek-V3.2-NVFP4 \
    --tp 4 \
    --quantization modelopt_fp4 \
    --moe-runner-backend flashinfer_trtllm \
    --kv-cache-dtype fp4_e2m1
```

## Research Sources

This summary was compiled from:
- Source code analysis of the SGLang repository
- Official documentation files
- Test files and benchmarks
- Pull Request discussions (#10078, #12612)

---

**Repository:** [yiliu30/sglang-fork](https://github.com/yiliu30/sglang-fork)  
**Original Repository:** [sgl-project/sglang](https://github.com/sgl-project/sglang)  
**Created:** February 2026
