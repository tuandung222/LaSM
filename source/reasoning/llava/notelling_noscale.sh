export CUDA_VISIBLE_DEVICES=2,3

# 参数配置
BEGIN_LAYER=12
END_LAYER=28
CHENG_NUM=1.0
INPUT_DIR="/data1/home/yanzihe/project/50samples/pop-up/attacked_llm_v2"
OUTPUT_DIR="/data1/home/yanzihe/project/50samples/results/0828/llava/notelling"
MODEL_PATH="/data2/models/llava-v1.6-vicuna-13b-hf"

# 运行主程序
python /data1/home/yanzihe/project/50samples/resoning/llava/notelling.py \
    --begin_layer $BEGIN_LAYER \
    --end_layer $END_LAYER \
    --cheng_num $CHENG_NUM \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --model_path $MODEL_PATH \
    --layer all