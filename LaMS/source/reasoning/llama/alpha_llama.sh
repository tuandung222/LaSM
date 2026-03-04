export CUDA_VISIBLE_DEVICES=0,1

BASE_OUTPUT_DIR="/data1/home/yanzihe/project/mllms_know/codes/llama/result/alpha_2"
SCRIPT_PATH="/data1/home/yanzihe/project/mllms_know/codes/llama/lams_llama.py"
INPUT_DIR="/data1/home/yanzihe/project/mllms_know/EnvDistraction_data/data/images/popupbox_phone_1b1i"
MODEL_PATH="/data2/models/Llama-3.2-11B-Vision-Instruct"
BEGIN_LAYER=12
END_LAYER=27

cheng=0.90
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
