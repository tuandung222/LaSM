export CUDA_VISIBLE_DEVICES=4,5

BASE_OUTPUT_DIR="/data1/home/yanzihe/project2026/200samples/scripts/atlas_hw/hw_result"
SCRIPT_PATH="/data1/home/yanzihe/project2026/200samples/scripts/atlas_hw/infer_main_hw.py"
# INPUT_DIR="/data1/home/yanzihe/project2026/200samples/popups/attacked_variants_from_bigjson_llm/new_large_induced_default/images"  # 记得替换成你实际的输入路径
INPUT_DIR="/data1/home/yanzihe/project/mllms_know/EnvDistraction_data/data/images/popupbox_phone_1b1i"  # 记得替换成你实际的输入路径
MODEL_PATH="/data4/hwveModels/atlas/hwgui_third_pad_dynamic_search_no_use_sop_atlas_v4/" #基于qwen微调，所以是28层
# ===== 固定 begin_layer ======
# 循环脚本
# BEGIN_LAYER=1  
# for (( END_LAYER=27; END_LAYER>BEGIN_LAYER; END_LAYER-- )); do
#   OUTPUT_DIR="${BASE_OUTPUT_DIR}/end_${END_LAYER}"

#   python "$SCRIPT_PATH" \
#     --input_dir "$INPUT_DIR" \
#     --output_dir "$OUTPUT_DIR" \
#     --begin_layer $BEGIN_LAYER \
#     --end_layer $END_LAYER \
#     --model_path $MODEL_PATH \
#     --cheng_num 1.1
# done

#===== 固定 end_layer ======

# END_LAYER=18
# for ((BEGIN_LAYER=5; END_LAYER>BEGIN_LAYER; BEGIN_LAYER++)); do
#   OUTPUT_DIR="${BASE_OUTPUT_DIR}/begin_${BEGIN_LAYER}"

#   python "$SCRIPT_PATH" \
#     --input_dir "$INPUT_DIR" \
#     --output_dir "$OUTPUT_DIR" \
#     --begin_layer $BEGIN_LAYER \
#     --end_layer $END_LAYER \
#     --model_path $MODEL_PATH \
#     --cheng_num 1.1 \
#     --json_path "/data1/home/yanzihe/project2026/200samples/popups/popup_train_jsons/new_large_induced_default_200.json"
# done


# ===== 单独脚本 =====
BEGIN_LAYER=7
OUTPUT_DIR="${BASE_OUTPUT_DIR}/end_${END_LAYER}"
python "$SCRIPT_PATH" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --begin_layer $BEGIN_LAYER \
  --end_layer 18 \
  --model_path $MODEL_PATH \
  --cheng_num 1.0 \
  # --json_path /data1/home/yanzihe/project2026/200samples/popups/popup_train_jsons/new_large_induced_default_200.json \