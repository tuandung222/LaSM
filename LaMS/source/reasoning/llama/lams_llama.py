import os
import json
import torch
import re
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import traceback
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration,MllamaForConditionalGeneration,AutoProcessor

# ==== 参数 ====
parser = argparse.ArgumentParser()
parser.add_argument("--layer", type=str, default="all")
parser.add_argument("--cheng_num", type=float, default=1.1)
parser.add_argument("--begin_layer", type=int, default=5)
parser.add_argument("--output_dir", type=str, default='/data1/home/yanzh/project/mllms_know/results')
parser.add_argument("--input_dir", type=str, default='/data1/home/yanzh/project/mllms_know/results')
parser.add_argument("--end_layer", type=int, default=15)
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()
print(args)

begin_num = args.begin_layer
end_num = args.end_layer
cheng_num = args.cheng_num

# ==== 路径设置 ====
cheng_str = str(cheng_num).replace('.', 'dot')
sub_dir = f"{begin_num}_{end_num}_{cheng_str}"
output_dir = os.path.join(args.output_dir, sub_dir)
image_folder = args.input_dir
os.makedirs(output_dir, exist_ok=True)

# ==== 放缩模型 ====
def scaling_llama(base_model, chat_model, begin_num, end_num, cheng_num):
    import copy
    import torch

    new_model = copy.deepcopy(base_model)

    with torch.no_grad():
        for i in range(begin_num, end_num):
            # 4.5以上版本
            # base_layer = new_model.language_model.layers[i]
            # ref_layer = chat_model.language_model.layers[i]
            
            # 4.48.3版本
            base_layer = new_model.language_model.model.layers[i]
            ref_layer = chat_model.language_model.model.layers[i]
            
            # ✅ 只处理含有 self_attn 的层（跳过 CrossAttentionDecoderLayer）
            if hasattr(base_layer, "self_attn") and hasattr(base_layer, "mlp"):
                # Attention 权重放缩
                base_layer.self_attn.q_proj.weight.copy_(ref_layer.self_attn.q_proj.weight * cheng_num)
                base_layer.self_attn.k_proj.weight.copy_(ref_layer.self_attn.k_proj.weight * cheng_num)
                base_layer.self_attn.v_proj.weight.copy_(ref_layer.self_attn.v_proj.weight * cheng_num)
                base_layer.self_attn.o_proj.weight.copy_(ref_layer.self_attn.o_proj.weight * cheng_num)

                # MLP 权重放缩
                base_layer.mlp.up_proj.weight.copy_(ref_layer.mlp.up_proj.weight * cheng_num)
                base_layer.mlp.gate_proj.weight.copy_(ref_layer.mlp.gate_proj.weight * cheng_num)
                base_layer.mlp.down_proj.weight.copy_(ref_layer.mlp.down_proj.weight * cheng_num)

            else:
                print(f"⏭️  跳过第 {i} 层（非 SelfAttention 层）")

    return new_model


# ==== 工具函数 ====
# def clean_answer(text: str) -> str:
#     if "assistant\n" in text:
#         text = text.split("assistant\n")[-1]
#     return text.strip()


def clean_answer(raw: str) -> str:
    """
    提取 'Answer: Button ...' 开头的回答。
    优先匹配格式 'Answer: Button <...>' 或 'Answer: Button xxx'，并去除句号。
    """
    match = re.search(r'Answer:\s*(Button\s+(?:<[^<>]+>|[\w\s]+))', raw)
    if match:
        return match.group(1).strip().rstrip('.')
    else:
        return "Unrecognized: no Button found"


def evaluate_accuracy(result_file: str, expected_answer="Button <icon-cross>"):
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = 0
    correct = 0
    predictions = []
    for item in data:
        pred = item.get("answer_1", "").strip()
        predictions.append(pred)
        total += 1
        if pred.lower() == expected_answer.lower():
            correct += 1
    acc = correct / total if total > 0 else 0
    print(f"总样本数: {total}")
    print(f"正确样本数: {correct}")
    print(f"准确率: {acc:.2%}")
    return predictions, total, correct, acc


def save_experiment_meta(save_path: str, test_name: str, sample_num: int, num_right_predict: int, accuracy: float, extra_info: dict = None):
    meta = {
        "test_name": test_name,
        "sample_num": sample_num,
        "num_right_predict": num_right_predict,
        "accuracy": round(accuracy, 4)
    }
    if extra_info:
        meta.update(extra_info)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"📄 实验信息已保存到: {save_path}")

# ==== Prompt ====
prompt_for_check = '''
You are given a screenshot. Your task is to find the exact visible button to click in order to buy an APPLE IPhone.

Return the button label in the following strict format:

Button <exact button text or icon label>

Examples:
Button <icon-cross>
Button Buy Now
Button Add to Cart
Button Subscribe

Now based on the image, return the matching button text or icon label strictly in this format.

'''

# ==== 模型加载 ====
device = 'cuda'
model_path = args.model_path

base_model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="eager",
    device_map="auto",
).eval()

chat_model = MllamaForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="eager",
    device_map="auto",
).eval()

model = scaling_llama(base_model, chat_model, begin_num, end_num, cheng_num)

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    model_path
)

# ==== 推理 ====
results = []
model.eval()

for filename in tqdm(sorted(os.listdir(image_folder))):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        continue
    image_path = os.path.join(image_folder, filename)

    try:
        image = Image.open(image_path).convert("RGB")

        # ✅ 改为官方格式构造 prompt
        text1 = f"<|image|><|begin_of_text|>{prompt_for_check} Answer:"

        inputs_1 = processor(
            image,
            text1,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            ids1 = model.generate(**inputs_1, max_new_tokens=50)
            raw1 = processor.batch_decode(ids1, skip_special_tokens=True)[0]
            answer_1 = clean_answer(raw1)

        print(answer_1)
        results.append({
            "img": filename,
            "img_path": image_path,
            "prompt_1": prompt_for_check,
            "answer_1": answer_1,
            "system": "default prompt template"
        })

    except Exception as e:
        print(f"❌ 跳过 {filename}：{e}")
        traceback.print_exc()
        continue


# ==== 保存 JSON ====
final_results = os.path.join(output_dir, "results.json")
with open(final_results, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ==== 准确率评估 ====
predictions, total, correct, acc = evaluate_accuracy(final_results)

# ==== 保存元信息 ====
meta_path = os.path.join(output_dir, "meta.json")
save_experiment_meta(
    save_path=meta_path,
    test_name="Qwen2.5-scaling-exp-no-attn",
    sample_num=total,
    num_right_predict=correct,
    accuracy=acc,
    extra_info={
        "cheng_num": cheng_num,
        "begin_layer": begin_num,
        "end_layer": end_num,
        "model_path": model_path,
        "image_folder": image_folder
    }
)
