export CUDA_VISIBLE_DEVICES=2,3

BASE_OUTPUT_DIR="/data1/home/yanzihe/project/mllms_know/codes/llama/result_0905"
SCRIPT_PATH="/data1/home/yanzihe/project/mllms_know/codes/llama/lams_llama.py"
INPUT_DIR="/data1/home/yanzihe/project/mllms_know/EnvDistraction_data/data/images/popupbox_phone_1b1i"  # 记得替换成你实际的输入路径
MODEL_PATH="/data2/models/Llama-3.2-11B-Vision-Instruct"

# ===== 固定 begin_layer ======
# BEGIN_LAYER=0
# for (( END_LAYER=28; END_LAYER>BEGIN_LAYER; END_LAYER--)); do
#   OUTPUT_DIR="${BASE_OUTPUT_DIR}/end_${END_LAYER}"

#   /data1/home/yanzihe/.conda/envs/llama/bin/python "$SCRIPT_PATH" \
#     --output_dir $OUTPUT_DIR \
#     --begin_layer $BEGIN_LAYER \
#     --end_layer $END_LAYER \
#     --model_path $MODEL_PATH \
#     --cheng_num 1.1 \
#     --json_path $JSON_PATH
# done

# ===== 固定 end_layer ======

# END_LAYER=12
# for ((BEGIN_LAYER=0; END_LAYER>BEGIN_LAYER; BEGIN_LAYER++)); do
#   OUTPUT_DIR="${BASE_OUTPUT_DIR}/begin_${BEGIN_LAYER}"

# python "$SCRIPT_PATH" \
#     --output_dir "$OUTPUT_DIR" \
#     --begin_layer $BEGIN_LAYER \
#     --end_layer $END_LAYER \
#     --model_path $MODEL_PATH \
#     --cheng_num  1.1 \
#     --json_path $JSON_PATH
# done

# ===== 寻找alpha =====
# for alpha in $(seq 0.80 0.05 1.30); do
#   OUTPUT_DIR="${BASE_OUTPUT_DIR}/alpha_$alpha"

# python "$SCRIPT_PATH" \
#     --input_dir "$INPUT_DIR" \
#     --output_dir "$OUTPUT_DIR" \
#     --begin_layer 9 \
#     --end_layer 12 \
#     --model_path $MODEL_PATH \
#     --cheng_num $alpha \
#     --json_path $JSON_PATH
# done


# for BEGIN_LAYER in {1..27}; do
#   OUTPUT_DIR="${BASE_OUTPUT_DIR}/begin_${BEGIN_LAYER}"
OUTPUT_DIR="${BASE_OUTPUT_DIR}/begin_${BEGIN_LAYER}"
python "$SCRIPT_PATH" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --begin_layer 12 \
  --end_layer 27 \
  --model_path $MODEL_PATH \
  --cheng_num 1.3

# ===== 单独脚本 =====
OUTPUT_DIR="${BASE_OUTPUT_DIR}/hw_result_single_test"
python "$SCRIPT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --begin_layer 9 \
  --end_layer 12 \
  --model_path $MODEL_PATH \
  --cheng_num 1.0 \
  --json_path $JSON_PATH