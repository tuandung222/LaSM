export CUDA_VISIBLE_DEVICES=2,3

BASE_OUTPUT_DIR="/data1/home/yanzihe/project/mllms_know/codes/os-atlas/results/alpha"
SCRIPT_PATH="/data1/home/yanzihe/project/mllms_know/codes/os-atlas/infer_main.py"
INPUT_DIR="/data1/home/yanzihe/project/mllms_know/EnvDistraction_data/data/images/popupbox_phone_1b1i"
MODEL_PATH="/data2/models/OS-Copilot/OS-Atlas-Pro-7B"
BEGIN_LAYER=15
END_LAYER=21

cheng=0.9
while (( $(echo "$cheng <= 1.40" | bc -l) )); do
  # 保留两位小数（防止 0.95000000000001 这种情况出现在路径中）
  cheng_str=$(printf "%.2f" $cheng)
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/cheng_cheng_${cheng_str}"
  mkdir -p "$OUTPUT_DIR"

  python "$SCRIPT_PATH" \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --begin_layer $BEGIN_LAYER \
    --end_layer $END_LAYER \
    --model_path "$MODEL_PATH" \
    --cheng_num $cheng_str

  cheng=$(echo "$cheng + 0.05" | bc)
done
