CUDA_VISIBLE_DEVICES=4,5 python /data1/home/yanzihe/project/paper1_lasm/source/reasoning/qwen2-vl/qwen2_12type.py \
    --model_path /data2/models/public/Qwen2-VL-7B-Instruct \
    --cheng_num 1.1 \
    --begin_layer 7 \
    --end_layer 18 \
    --layer -1 \
    --output_dir /data1/home/yanzihe/project/paper1_lasm/result \
    --input_dir /data1/home/yanzihe/project/50samples/pop-up/attacked_llm_v2 \
    --json_path /data1/home/yanzihe/project/50samples/pop-up/attacked_llm_v2/metadata_filled.json