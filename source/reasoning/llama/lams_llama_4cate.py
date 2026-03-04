# lams_llama.py
import os
import re
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import traceback
from transformers import MllamaForConditionalGeneration, AutoProcessor

# ============ 环境 ============
# 若你有固定卡位需求，可取消下一行注释
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# ============ 参数 ============
parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=str, default="all")
parser.add_argument("--cheng_num", type=float, default=1.15)
parser.add_argument("--begin_layer", type=int, default=12)
parser.add_argument("--end_layer", type=int, default=28)

parser.add_argument("--model_path", type=str,default="/data2/models/Llama-3.2-11B-Vision-Instruct")
parser.add_argument("--jsonl_path", type=str, default="/data1/home/yanzihe/project/50samples/results_vis/data/loc/output_category_4.jsonl")
parser.add_argument("--input_dir", type=str, default="/data1/home/yanzihe/project/50samples/results_vis/data/images/category_4")  # 放截图的根目录
parser.add_argument("--output_dir", type=str, default="/data1/home/yanzihe/project/mllms_know/results_llama")
args = parser.parse_args()
print(args)

begin_num = args.begin_layer
end_num = args.end_layer
cheng_num = args.cheng_num

# ============ 输出路径 ============
cheng_str = str(cheng_num).replace('.', 'dot')
sub_dir = f"{begin_num}_{end_num}_{cheng_str}"
output_dir = os.path.join(args.output_dir, f"category_4_{sub_dir}")
os.makedirs(output_dir, exist_ok=True)

def save_raw_output(raw: str) -> str:
    """只保存Answer:后的实际输出部分"""
    # 用正则找到Answer:后的所有内容
    m = re.search(r"\nAnswer:\s*(.*?)$", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    return raw

# ============ 放缩（LaSM） ============
def scaling_llama(base_model, chat_model, begin_num, end_num, cheng_num):
    import copy
    new_model = copy.deepcopy(base_model)
    with torch.no_grad():
        for i in range(begin_num, end_num):
            # 4.48.3 版本路径
            base_layer = new_model.language_model.model.layers[i]
            ref_layer = chat_model.language_model.model.layers[i]

            if hasattr(base_layer, "self_attn") and hasattr(base_layer, "mlp"):
                # Attention
                base_layer.self_attn.q_proj.weight.copy_(ref_layer.self_attn.q_proj.weight * cheng_num)
                base_layer.self_attn.k_proj.weight.copy_(ref_layer.self_attn.k_proj.weight * cheng_num)
                base_layer.self_attn.v_proj.weight.copy_(ref_layer.self_attn.v_proj.weight * cheng_num)
                base_layer.self_attn.o_proj.weight.copy_(ref_layer.self_attn.o_proj.weight * cheng_num)
                # MLP
                base_layer.mlp.up_proj.weight.copy_(ref_layer.mlp.up_proj.weight * cheng_num)
                base_layer.mlp.gate_proj.weight.copy_(ref_layer.mlp.gate_proj.weight * cheng_num)
                base_layer.mlp.down_proj.weight.copy_(ref_layer.mlp.down_proj.weight * cheng_num)
            else:
                print(f"⏭️  跳过第 {i} 层（非 SelfAttention 层）")
    return new_model

# ============ 工具函数 ============
def build_prompt(goal: str) -> str:
    """
    让模型只在 1/2/3/4 中选一个:
    1=左上, 2=右上, 3=左下, 4=右下
    """
    return (
        "You are given a shopping screenshot that shows exactly four results arranged in a 2x2 grid.\n"
        "The mapping is: 1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right.\n"
        f"User goal: {goal}\n"
        "Choose the SINGLE best result that answers the goal. "
        "Return strictly in the format: Answer: 1 or Answer: 2 or Answer: 3 or Answer: 4.\n"
        "Do not output anything else."
    )

def parse_choice_from_text(text: str) -> str:
    """
    从模型输出解析出1/2/3/4。
    优先返回Answer:后的第一个数字。
    如果找不到明确的答案格式，返回'unknown'。
    """
    # 优先匹配 "Answer: N" 格式后的第一个数字
    m = re.search(r"\nAnswer:.*?([1-4])", text)
    if m:
        return m.group(1)
        
    # 其次匹配独立的数字1-4
    m2 = re.search(r"\b([1-4])\b", text) 
    if m2:
        return m2.group(1)
        
    # 找不到任何合法答案时返回unknown
    return "unknown"

def extract_indices_from_label_items(items):
    """
    items 形如：
      ["Click result 1, title...", [x1,y1,x2,y2]]
    返回编号列表: [1, ...]
    """
    idxs = []
    for it in items:
        try:
            s = it[0] if isinstance(it, list) and len(it) > 0 else ""
            m = re.search(r"result\s+([1-4])", s)
            if m:
                idxs.append(int(m.group(1)))
        except Exception:
            pass
    return idxs

def infer_image_path(modified_file: str, root: str) -> str:
    """
    由 modified_file 推断截图路径：
    比如 category_4_html_xxx.html -> root/category_4_html_xxx.png/.jpg/.jpeg
    """
    base = os.path.splitext(modified_file)[0]
    candidates = [
        os.path.join(root, base + ".png"),
        os.path.join(root, base + ".jpg"),
        os.path.join(root, base + ".jpeg"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return ""  # 找不到则返回空

# ============ 加载模型 ============
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = args.model_path

base_model = MllamaForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float16, attn_implementation="eager", device_map="auto"
).eval()

chat_model = MllamaForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float16, attn_implementation="eager", device_map="auto"
).eval()

model = scaling_llama(base_model, chat_model, begin_num, end_num, cheng_num)
processor = AutoProcessor.from_pretrained(model_path)

# ============ 读取 category_4 样本 ============
samples = []
with open(args.jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("target") != "category_4":
            continue
        samples.append(item)

print(f"Loaded category_4 samples: {len(samples)}")

# ============ 推理 ============
results = []
total = 0
correct = 0
wrong_bad = 0  # 专门统计命中 bad 的错误数

for ex in tqdm(samples, desc="Inferencing"):
    goal = ex.get("goal", "").strip()
    modified_file = ex.get("modified_file", "")
    img_path = infer_image_path(modified_file, args.input_dir)

    if not img_path:
        print(f"⚠️ 未找到截图，跳过：{modified_file}")
        continue

    # 解析 gold/bad 允许的编号
    label = ex.get("label", {})
    gold_ids = set(extract_indices_from_label_items(label.get("gold", [])))
    bad_ids = set(extract_indices_from_label_items(label.get("bad", [])))

    try:
        image = Image.open(img_path).convert("RGB")
        prompt = build_prompt(goal)
        text = f"<|image|><|begin_of_text|>{prompt}\nAnswer:"

        inputs = processor(image, text, add_special_tokens=False, return_tensors="pt").to(device)

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=100)
            raw = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            pred_choice = parse_choice_from_text(raw)

        is_correct = False
        hit_bad = False
        numeric_pred = None
        if pred_choice in {"1", "2", "3", "4"}:
            numeric_pred = int(pred_choice)
            if numeric_pred in gold_ids:
                is_correct = True
            elif numeric_pred in bad_ids:
                hit_bad = True
                is_correct = False
            else:
                is_correct = False
        else:
            is_correct = False

        total += 1
        if is_correct:
            correct += 1
        if hit_bad:
            wrong_bad += 1

        results.append({
            "modified_file": modified_file,
            "img_path": img_path,
            "goal": goal,
            "gold_ids": sorted(list(gold_ids)),
            "bad_ids": sorted(list(bad_ids)),
            "prompt": prompt,
            "raw_output": save_raw_output(raw),  # 只保存Answer:后的内容,
            "pred_choice": pred_choice,
            "pred_numeric": numeric_pred,
            "is_correct": is_correct,
            "hit_bad": hit_bad,
            "system": f"category_4_prompt_range_{begin_num}_{end_num}_alpha_{cheng_num}",
        })

    except Exception as e:
        print(f"❌ 推理失败：{modified_file} | {e}")
        traceback.print_exc()
        continue

# ============ 保存结果与评估 ============
final_results = os.path.join(output_dir, "results_category4.json")
with open(final_results, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

acc = (correct / total) if total > 0 else 0.0
meta = {
    "test_name": "LaSM-category_4",
    "sample_num": total,
    "num_right_predict": correct,
    "accuracy": round(acc, 4),
    "wrong_bad": wrong_bad,
    "cheng_num": cheng_num,
    "begin_layer": begin_num,
    "end_layer": end_num,
    "model_path": model_path,
    "jsonl_path": args.jsonl_path,
    "image_root": args.input_dir,
}
with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print("\n====== SUMMARY ======")
print(f"总样本数: {total}")
print(f"正确样本数: {correct}")
print(f"准确率  : {acc:.2%}")
print(f"命中 bad 的错误数: {wrong_bad}")
print(f"结果文件: {final_results}")
print(f"元信息  : {os.path.join(output_dir, 'meta.json')}")
