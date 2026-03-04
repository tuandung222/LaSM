export CUDA_VISIBLE_DEVICES=0,1

BASE_OUTPUT_DIR="/data1/home/yanzh/project/mllms_know/codes/os-atlas/results/OS-Atlas-Pro-7B-0619"
SCRIPT_PATH="/data1/home/yanzh/project/mllms_know/codes/os-atlas/infer_main.py"
INPUT_DIR="/data1/home/yanzh/project/mllms_know/EnvDistraction_data/data/images/popupbox_phone_1b1i"  # 记得替换成你实际的输入路径
MODEL_PATH="/data1/models/OS-Atlas-Pro-7B"
END_LAYER=21  # 固定 begin_layer
BEGIN_LAYER=0
for (( BEGIN_LAYER=0; END_LAYER>BEGIN_LAYER; BEGIN_LAYER++ )); do
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/end_${END_LAYER}"
    python "$SCRIPT_PATH" \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --begin_layer $BEGIN_LAYER \
        --end_layer $END_LAYER \
        --model_path $MODEL_PATH \
        --cheng_num 1.0
done