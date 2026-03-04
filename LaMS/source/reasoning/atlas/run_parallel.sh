#!/usr/bin/env bash
# GPU0 和 GPU3 并行执行，各自串行执行三个推理任务
set -euo pipefail

PY="/data1/home/yanzihe/project/50samples/resoning/atlas/infer_main_atlas.py"
OUT="/data1/home/yanzihe/project/50samples/results/cvpr_telling"
MODEL="/data4/hwveModels/atlas/hwgui_third_pad_dynamic_search_no_use_sop_atlas_v4"
BEGIN=15
END=21

# GPU0: scaled
(
  echo ">>> GPU0 down scaled"
  CUDA_VISIBLE_DEVICES=0,1 python "$PY" \
      --input_dir "/data1/home/yanzihe/project/50samples/pop-up/attacked_llm_1105/down" \
      --output_dir "$OUT" --begin_layer "$BEGIN" --end_layer "$END" \
      --model_path "$MODEL" --cheng_num 1.1 \
      --json_path "/data1/home/yanzihe/project/50samples/pop-up/attacked_llm_1105/down_metadata.json"

  echo ">>> GPU0 mid scaled"
  CUDA_VISIBLE_DEVICES=0,1 python "$PY" \
      --input_dir "/data1/home/yanzihe/project/50samples/pop-up/attacked_llm_1105/mid" \
      --output_dir "$OUT" --begin_layer "$BEGIN" --end_layer "$END" \
      --model_path "$MODEL" --cheng_num 1.1 \
      --json_path "/data1/home/yanzihe/project/50samples/pop-up/attacked_llm_1105/mid_metadata.json"

  echo ">>> GPU0 up scaled"
  CUDA_VISIBLE_DEVICES=0,1 python "$PY" \
      --input_dir "/data1/home/yanzihe/project/50samples/pop-up/attacked_llm_1105/up" \
      --output_dir "$OUT" --begin_layer "$BEGIN" --end_layer "$END" \
      --model_path "$MODEL" --cheng_num 1.1 \
      --json_path "/data1/home/yanzihe/project/50samples/pop-up/attacked_llm_1105/up_metadata.json"

  echo "GPU0 完成 ✅"
) &


# GPU3: noscale
(
  echo ">>> GPU3 mid noscale"
  CUDA_VISIBLE_DEVICES=2,3 python "$PY" \
      --input_dir "/data1/home/yanzihe/project/50samples/pop-up/attacked_llm_1105_noscale/mid" \
      --output_dir "$OUT" --begin_layer "$BEGIN" --end_layer "$END" \
      --model_path "$MODEL" --cheng_num 1.0 \
      --json_path "/data1/home/yanzihe/project/50samples/pop-up/attacked_llm_1105/mid_metadata.json"

  echo ">>> GPU3 up noscale"
  CUDA_VISIBLE_DEVICES=2,3 python "$PY" \
      --input_dir "/data1/home/yanzihe/project/50samples/pop-up/attacked_llm_1105_noscale/up" \
      --output_dir "$OUT" --begin_layer "$BEGIN" --end_layer "$END" \
      --model_path "$MODEL" --cheng_num 1.0 \
      --json_path "/data1/home/yanzihe/project/50samples/pop-up/attacked_llm_1105/up_metadata.json"

  echo ">>> GPU3 down noscale"
  CUDA_VISIBLE_DEVICES=2,3 python "$PY" \
      --input_dir "/data1/home/yanzihe/project/50samples/pop-up/attacked_llm_1105_noscale/down" \
      --output_dir "$OUT" --begin_layer "$BEGIN" --end_layer "$END" \
      --model_path "$MODEL" --cheng_num 1.0 \
      --json_path "/data1/home/yanzihe/project/50samples/pop-up/attacked_llm_1105/down_metadata.json"

  echo "GPU3 完成 ✅"
) &


wait
echo "🎯 全部任务执行完毕！"
