import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import traceback
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, MllamaForConditionalGeneration, AutoProcessor

# ==== Parameters ====
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=1.1)
parser.add_argument("--lower_bound", type=int, default=5)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--input_dir", type=str)
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

# ==== Scaling Model ====
def scaling_llama(base_model, chat_model, begin_num, end_num, alpha):
    import copy
    import torch

    new_model = copy.deepcopy(base_model)

    with torch.no_grad():
        for i in range(begin_num, end_num):
            base_layer = new_model.language_model.model.layers[i]
            ref_layer = chat_model.language_model.model.layers[i]

            # ✅ Only process layers with self_attn (skip CrossAttentionDecoderLayer)
            if hasattr(base_layer, "self_attn") and hasattr(base_layer, "mlp"):
                # Scale Attention weights
                base_layer.self_attn.q_proj.weight.copy_(ref_layer.self_attn.q_proj.weight * alpha)
                base_layer.self_attn.k_proj.weight.copy_(ref_layer.self_attn.k_proj.weight * alpha)
                base_layer.self_attn.v_proj.weight.copy_(ref_layer.self_attn.v_proj.weight * alpha)
                base_layer.self_attn.o_proj.weight.copy_(ref_layer.self_attn.o_proj.weight * alpha)

                # Scale MLP weights
                base_layer.mlp.up_proj.weight.copy_(ref_layer.mlp.up_proj.weight * alpha)
                base_layer.mlp.gate_proj.weight.copy_(ref_layer.mlp.gate_proj.weight * alpha)
                base_layer.mlp.down_proj.weight.copy_(ref_layer.mlp.down_proj.weight * alpha)

            else:
                print(f"⏭️  Skipping layer {i} (not a SelfAttention layer)")

    return new_model


# ==== Utility Functions ====
def clean_answer(text: str) -> str:
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1]
    return text.strip()

def evaluate_accuracy(result_file: str, expected_answer="Button <icon-close>"):
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
    print(f"Correct samples: {correct}")
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
Button <icon-close>
Button Buy Now
Button Add to Cart
Button Subscribe

Now based on the image, return the matching button text or icon label strictly in this format.

Anser:
'''

# ==== Load Model ====
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

model = scaling_llama(base_model, chat_model, begin_num, end_num, alpha)
processor = AutoProcessor.from_pretrained(
    model_path
)

# ==== Inference ====
results = []
model.eval()

for filename in tqdm(sorted(os.listdir(image_folder))):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        continue
    image_path = os.path.join(image_folder, filename)

    try:
        # Correctly load the image
        image = Image.open(image_path).convert("RGB")

        # Construct prompt
        msg1 = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_for_check + " Answer:"}]}]
        text1 = processor.apply_chat_template(msg1, add_generation_prompt=True)

        inputs_1 = processor(
            image,
            text1,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(device)

        # Inference
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
        print(f"❌skip {filename}：{e}")
        traceback.print_exc()
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
    test_name="llama-scaling-exp-no-attn",
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
