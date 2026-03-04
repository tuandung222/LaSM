import os
import json
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import  AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# ==== 参数 ====
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str,  help="model path")
parser.add_argument("--layer", type=str, default="all", help="指定 attention 层号，-1 表示最后一层，all 表示全部层")
parser.add_argument("--cheng_num", type=float, default=1.1, help="放缩系数 cheng_num")
parser.add_argument("--begin_layer", type=int, default=5, help="放缩起始层编号")
parser.add_argument("--output_dir", type=str, default='/data1/home/yanzh/project/mllms_know/results', help="存储路径")
parser.add_argument("--input_dir", type=str, default='/data1/home/yanzh/project/mllms_know/results', help="图片路径")
parser.add_argument("--end_layer", type=int, default=15, help="放缩结束层编号")
args = parser.parse_args()
print(args)

begin_num=args.begin_layer
end_num=args.end_layer
cheng_num=args.cheng_num


# ==== 路径设置 ====

cheng_str = str(cheng_num).replace('.', 'dot')
sub_dir = f"{begin_num}_{end_num}_{cheng_str}"
output_dir = os.path.join(args.output_dir, sub_dir)

device = 'cuda'
model_path = args.model_path
model_name= os.path.basename(model_path)

image_folder = args.input_dir
attn_dir = os.path.join(output_dir, 'attn_maps')
attn_right_dir = os.path.join(attn_dir, "right")
attn_wrong_dir = os.path.join(attn_dir, "wrong")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(attn_dir, exist_ok=True)
os.makedirs(attn_right_dir, exist_ok=True)
os.makedirs(attn_wrong_dir, exist_ok=True)

# ==== 模型 & 处理器 ====
## 放缩系数
def scaling_qwen2vl(base_model, chat_model, begin_num, end_num, cheng_num):
    import copy
    new_model = copy.deepcopy(base_model)
    with torch.no_grad():
        for i in range(begin_num, end_num):
            # Attention 权重放缩
            new_model.model.layers[i].self_attn.q_proj.weight.copy_(
                chat_model.model.layers[i].self_attn.q_proj.weight * cheng_num)
            new_model.model.layers[i].self_attn.k_proj.weight.copy_(
                chat_model.model.layers[i].self_attn.k_proj.weight * cheng_num)
            new_model.model.layers[i].self_attn.v_proj.weight.copy_(
                chat_model.model.layers[i].self_attn.v_proj.weight * cheng_num)
            new_model.model.layers[i].self_attn.o_proj.weight.copy_(
                chat_model.model.layers[i].self_attn.o_proj.weight * cheng_num)

            # MLP 权重放缩
            new_model.model.layers[i].mlp.up_proj.weight.copy_(
                chat_model.model.layers[i].mlp.up_proj.weight * cheng_num)
            new_model.model.layers[i].mlp.gate_proj.weight.copy_(
                chat_model.model.layers[i].mlp.gate_proj.weight * cheng_num)
            new_model.model.layers[i].mlp.down_proj.weight.copy_(
                chat_model.model.layers[i].mlp.down_proj.weight * cheng_num)
    return new_model




# ==== 工具函数 ====
def clean_answer(text: str) -> str:
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1]
    return text.strip()

def calculate_plt_size(n):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols

def save_experiment_meta(
    save_path: str,
    test_name: str,
    sample_num: int,
    num_right_predict: int,
    accuracy: float,
    extra_info: dict = None,
):
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

def save_attention_maps(attns, general_attns, pos, pos_end, output_shape, img_base_name, save_dir):
    paths = []
    heatmap_save_dir = os.path.join(save_dir, "heatmap_values", img_base_name)
    os.makedirs(heatmap_save_dir, exist_ok=True)
    total_layers = len(attns)
    if args.layer == "all":
        rows, cols = calculate_plt_size(total_layers)
        fig, axes = plt.subplots(rows, cols, figsize=(10.8, 16))
        for i, ax in enumerate(axes.flatten()):
            if i < total_layers:
                att = attns[i][0, :, -1, pos:pos_end].mean(dim=0).cpu().numpy()
                general_att = general_attns[i][0, :, -1, pos:pos_end].mean(dim=0).cpu().numpy()
                heat = att / (general_att + 1e-6)
                ax.imshow(heat.reshape(output_shape), cmap="viridis", interpolation="nearest")
                ax.set_title(f"Layer {i+1}")
                ax.axis("off")
            else:
                ax.axis("off")
        plt.tight_layout()
        path = os.path.join(save_dir, f"{img_base_name}_2.5attn_all_layers.png")
        plt.savefig(path, dpi=300)
        plt.close()
        paths.append(path)
    else:
        idx = int(args.layer)
        if idx < 0:
            idx = total_layers + idx
        att = attns[idx][0, :, -1, pos:pos_end].mean(dim=0).cpu().numpy() # attns[idx]范围0-27
        general_att = general_attns[idx][0, :, -1, pos:pos_end].mean(dim=0).cpu().numpy()
        heat = att / (general_att + 1e-6)
        plt.figure(figsize=(5, 5))
        plt.imshow(heat.reshape(output_shape), cmap="viridis", interpolation="nearest")
        plt.axis("off")
        plt.tight_layout()
        path = os.path.join(save_dir, f"{img_base_name}_attn_layer{idx+1}.png") # 我们输出没有说第0层，而是第1层开始的
        plt.savefig(path, dpi=300)
        plt.close()
        paths.append(path)
        
        npy_path = os.path.join(heatmap_save_dir, f"layer_{idx+1}.npy")
        np.save(npy_path, heat.reshape(output_shape))

    return paths

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
    print(f"总样本数: {total}")
    print(f"正确样本数: {correct}")
    print(f"准确率: {acc:.2%}")
    return predictions, total, correct, acc

# ==== 主逻辑 ====
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



base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="eager",
    device_map="auto",
).eval()

chat_model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    attn_implementation="eager",
    device_map="auto",
).eval()

# 确定放缩的层数

model = scaling_qwen2vl(base_model, chat_model, begin_num, end_num, cheng_num)

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
    padding_side='left',
    use_fast=True,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)


results = []
prompt_1 = prompt_for_check
prompt_2 = "Write a general description of the image."
model.eval()

for filename in tqdm(sorted(os.listdir(image_folder))):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        continue
    image_path = os.path.join(image_folder, filename)

    try:
        # prompt 1
        msg1 = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_1 + " Answer:"}]}]
        image_inputs, _ = process_vision_info(msg1)
        text1 = processor.apply_chat_template(msg1, tokenize=False, add_generation_prompt=True)
        inputs_1 = processor(text=[text1], images=image_inputs, padding=True, return_tensors="pt").to(device)

        # prompt 2
        msg2 = [{"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_2 + " Answer:"}]}]
        text2 = processor.apply_chat_template(msg2, tokenize=False, add_generation_prompt=True)
        inputs_2 = processor(text=[text2], images=image_inputs, padding=True, return_tensors="pt").to(device)

        # 生成回答
        with torch.no_grad():
            ids1 = model.generate(**inputs_1, max_new_tokens=500)
            ids2 = model.generate(**inputs_2, max_new_tokens=500)
            raw1 = processor.batch_decode(ids1, skip_special_tokens=True)[0]
            raw2 = processor.batch_decode(ids2, skip_special_tokens=True)[0]
            answer_1 = clean_answer(raw1)
            answer_2 = clean_answer(raw2)

        print(answer_1)

        # 注意力提取
        with torch.no_grad():
            output1 = model(**inputs_1, output_attentions=True)
            output2 = model(**inputs_2, output_attentions=True)

        input_ids = inputs_1["input_ids"].tolist()[0]
        vision_start_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        vision_end_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
        pos = input_ids.index(vision_start_id) + 1
        pos_end = input_ids.index(vision_end_id)

        aux = processor.image_processor(images=image_inputs)
        output_shape = aux["image_grid_thw"].squeeze(0)[1:] // 2
        output_shape = output_shape.astype(int)

        is_close = answer_1.strip().lower() == "Button <icon-cross>"
        attn_save_dir = attn_right_dir if is_close else attn_wrong_dir
        attn_paths = save_attention_maps(output1.attentions, output2.attentions, pos, pos_end, output_shape, os.path.splitext(filename)[0], attn_save_dir)

        results.append({
            "img": filename,
            "img_path": image_path,
            "prompt_1": prompt_1,
            "prompt_2": prompt_2,
            "answer_1": answer_1,
            "answer_2": answer_2,
            "attn_path": attn_paths,
            "model": model_name
        })

    except Exception as e:
        print(f"❌ 跳过 {filename}：{e}")
        continue

# ==== 保存 JSON ====
final_results = os.path.join(output_dir, "0417_results.json")
with open(final_results, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 评估 & 保存元信息
predictions, total, correct, acc = evaluate_accuracy(final_results)

meta_path = os.path.join(output_dir, "meta.json")
save_experiment_meta(
    save_path=meta_path,
    test_name="Qwen2-scaling-exp-0421",
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