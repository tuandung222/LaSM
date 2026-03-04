export CUDA_VISIBLE_DEVICES=1,2

# 参数配置
BEGIN_LAYER=7
END_LAYER=18
CHENG_NUM=1.1
INPUT_DIR="/data1/home/yanzihe/project/50samples/pop-up/attacked_llm_v2"
OUTPUT_DIR="/data1/home/yanzihe/project/50samples/results/0828/qwen/telling"
MODEL_PATH="/data2/models/Qwen2-VL-7B-Instruct"

# 运行主程序
python /data1/home/yanzihe/project/paper1_lasm/source/reasoning/qwen2-vl/qwen2_Ma.py \
    --begin_layer $BEGIN_LAYER \
    --end_layer $END_LAYER \
    --cheng_num $CHENG_NUM \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --model_path $MODEL_PATH \
    --layer all