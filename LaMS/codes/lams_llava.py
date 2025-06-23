import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

# ==== Parameters ====
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=1.1)
parser.add_argument("--lower_bound", type=int, default=5)
parser.add_argument("--output_dir", type=str, default='./results')
parser.add_argument("--input_dir", type=str, default='./imgs')
parser.add_argument("--upper_bound", type=int, default=15)
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()
print(args)

begin_num = args.lower_bound
end_num = args.upper_bound
alpha = args.alpha

# ==== Path Configuration ====
cheng_str = str(alpha).replace('.', 'dot')
sub_dir = f"{begin_num}_{end_num}_{cheng_str}"
output_dir = os.path.join(args.output_dir, sub_dir)
image_folder = args.input_dir
os.makedirs(output_dir, exist_ok=True)

# ==== Scale Model ====
def scaling_llava(base_model, chat_model, begin_num, end_num, alpha):
    import copy
    new_model = copy.deepcopy(base_model)
    with torch.no_grad():
        for i in range(begin_num, end_num):
            text_base = new_model.language_model.model.layers[i]
            text_chat = chat_model.language_model.model.layers[i]
            text_base.self_attn.q_proj.weight.copy_(text_chat.self_attn.q_proj.weight * alpha)
            text_base.self_attn.k_proj.weight.copy_(text_chat.self_attn.k_proj.weight * alpha)
            text_base.self_attn.v_proj.weight.copy_(text_chat.self_attn.v_proj.weight * alpha)
            text_base.self_attn.o_proj.weight.copy_(text_chat.self_attn.o_proj.weight * alpha)
            text_base.mlp.gate_proj.weight.copy_(text_chat.mlp.gate_proj.weight * alpha)
            text_base.mlp.up_proj.weight.copy_(text_chat.mlp.up_proj.weight * alpha)
            text_base.mlp.down_proj.weight.copy_(text_chat.mlp.down_proj.weight * alpha)
    return new_model

# ==== Utility Functions ====
def clean_answer(text: str) -> str:
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1]
    return text.strip()

def evaluate_accuracy(result_file: str, expected_answer="Button <icon-cross>"):
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = 0
    correct = 0
    predictions = []
    for item in data:
        raw_output = item.get("answer_1", "")
        pred = clean_answer(raw_output)
        predictions.append(pred)
        total += 1
        if pred.lower() == expected_answer.lower():
            correct += 1
    acc = correct / total if total > 0 else 0
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {acc:.2%}")
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
    print(f"📄 Experiment metadata saved to: {save_path}")

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

Now based on the image, return the matching button text or icon label in this format.

Answer:
'''

# ==== Load Model ====
device = 'cuda'
model_path = args.model_path

base_model = LlavaNextForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="eager",
    device_map="auto",
).eval()

chat_model = LlavaNextForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="eager",
    device_map="auto",
).eval()

model = scaling_llava(base_model, chat_model, begin_num, end_num, alpha)

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = LlavaNextProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    padding_side='left',
    use_fast=True,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

# ==== Inference ====
results = []
model.eval()

for filename in tqdm(sorted(os.listdir(image_folder))):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        continue
    image_path = os.path.join(image_folder, filename)

    try:
        msg1 = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_for_check + " Answer:"}]}]
        image_inputs = Image.open(image_path).convert("RGB")
        text1 = processor.apply_chat_template(msg1, tokenize=False, add_generation_prompt=True)
        inputs_1 = processor(text=[text1], images=image_inputs, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            ids1 = model.generate(**inputs_1, max_new_tokens=500)
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
        print(f"❌ Skipping {filename}: {e}")
        continue

# ==== Save JSON ====
final_results = os.path.join(output_dir, "results.json")
with open(final_results, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ==== Accuracy Evaluation ====
predictions, total, correct, acc = evaluate_accuracy(final_results)

# ==== Save Metadata ====
meta_path = os.path.join(output_dir, "meta.json")
save_experiment_meta(
    save_path=meta_path,
    test_name="llava-scaling-exp-no-attn",
    sample_num=total,
    num_right_predict=correct,
    accuracy=acc,
    extra_info={
        "alpha": alpha,
        "lower_bound": begin_num,
        "upper_bound": end_num,
        "model_path": model_path,
        "image_folder": image_folder
    }
)
