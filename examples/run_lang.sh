# SGLANG_HPU_SKIP_WARMUP=true \
# SGLANG_INC_QUANT_CONFIG="inc_measure.json" \
# python runtime/engine/offline_batch_inference.py \
#     --device hpu \
#     --model-path  /mnt/weka/data/pytorch/llama3.2/Meta-Llama-3.2-1B/ \
#     --tp-size 1 \
#     --dtype bfloat16
SGLANG_HPU_SKIP_WARMUP=true \
SGLANG_INC_QUANT_CONFIG="inc_quant.json" \
python runtime/engine/offline_batch_inference.py \
    --device hpu \
    --model-path  /mnt/weka/data/pytorch/llama3.2/Meta-Llama-3.2-1B/ \
    --tp-size 1 \
    --dtype bfloat16