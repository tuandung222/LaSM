export CUDA_VISIBLE_DEVICES=0

BASE_OUTPUT_DIR="/data1/home/yanzh/project/mllms_know/codes/lams_llama/result"
SCRIPT_PATH="/data1/home/yanzh/project/mllms_know/codes/llama/lams_llama.py"
INPUT_DIR="/data1/home/yanzh/project/mllms_know/EnvDistraction_data/data/images/popupbox_phone_1b1i"  # 记得替换成你实际的输入路径
MODEL_PATH="/data1/home/yanzh/model/Llama-3.2-11B-Vision-Instruct"
# BEGIN_LAYER=0  # 固定 begin_layer

# for (( END_LAYER=34; END_LAYER>BEGIN_LAYER; END_LAYER-- )); do # llava有40层
#   OUTPUT_DIR="${BASE_OUTPUT_DIR}/end_${END_LAYER}"

for BEGIN_LAYER in {1..27}; do
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/begin_${BEGIN_LAYER}"

  python "$SCRIPT_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --begin_layer $BEGIN_LAYER \
    --end_layer 27 \
    --model_path $MODEL_PATH \
    --cheng_num 1.1
done