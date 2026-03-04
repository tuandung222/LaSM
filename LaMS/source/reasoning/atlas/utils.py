
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import re
import os

def clean_answer(cmd: str):
    return cmd.replace("actions:\n", "").replace("<|im_end|>", "").strip()


def extract_user_prompt(sample):
    """
    从样本中提取 user 的文字 prompt。
    兼容以下情况：
    1. sample["content"] 存在（旧格式）
    2. sample["messages"] 存在，且其中 role == "user"
    3. content 可能是字符串或列表（image 对象）
    """
    if "content" in sample:
        return str(sample["content"]).strip()

    if "messages" in sample and isinstance(sample["messages"], list):
        for msg in sample["messages"]:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                # 若 content 是字符串
                if isinstance(content, str):
                    return content.strip()
                # 若 content 是列表（通常是图片内容）
                elif isinstance(content, list):
                    # 跳过纯 image 类型
                    text_parts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
                    if text_parts:
                        return "\n".join(text_parts).strip()
        # 如果没有纯文字 user，则返回空
        return ""
    return ""

def calculate_plt_size(n):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols

def extract_coords(cmd: str):
    return [tuple(map(int, m)) for m in re.findall(r"\[(\d+),\s*(\d+)\]", cmd)]

def coords_close(pred_coords, gold_coords, model_size=1000, ratio=0.14):
    if len(pred_coords) != len(gold_coords):
        return False
    tol = model_size * ratio
    for (px, py), (gx, gy) in zip(pred_coords, gold_coords):
        if abs(px - gx) > tol or abs(py - gy) > tol:
            return False
    return True


def evaluate_accuracy(result_file: str, model_size=1000, ratio=0.14):
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    total = 0
    correct = 0
    predictions = []
    for item in data:
        raw_output = item.get("answer_1", "")
        pred_coords = extract_coords(raw_output)
        predictions.append(pred_coords)
        gold_coords=item.get("gold_coords", "")
        total += 1
        if coords_close(pred_coords, gold_coords, model_size, ratio):
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

def save_attention_maps(attns, general_attns, pos, pos_end, output_shape, img_base_name, save_dir, args_layer="all"):
    paths = []
    heatmap_save_dir = os.path.join(save_dir, "heatmap_values", img_base_name)
    os.makedirs(heatmap_save_dir, exist_ok=True)
    total_layers = len(attns)
    if args_layer == "all":
        rows, cols = calculate_plt_size(total_layers)
        fig, axes = plt.subplots(rows, cols, figsize=(10.8, 16))
        for i, ax in enumerate(axes.flatten()):
            if i < total_layers:
                att = attns[i][0, :, -1, pos:pos_end].mean(dim=0).cpu().numpy()
                general_att = general_attns[i][0, :, -1, pos:pos_end].mean(dim=0).cpu().numpy()
                min_len = min(len(att), len(general_att))
                att = att[:min_len]
                general_att = general_att[:min_len]
                heat = att / (general_att + 1e-6)
                ax.imshow(heat.reshape(output_shape), cmap="viridis", interpolation="nearest")
                ax.set_title(f"Layer {i+1}")
                ax.axis("off")
                
                npy_path = os.path.join(heatmap_save_dir, f"layer_{i+1}.npy")
                np.save(npy_path, heat.reshape(output_shape))   
            else:
                ax.axis("off")
            
        plt.tight_layout()
        path = os.path.join(save_dir, f"{img_base_name}_attn_all_layers.png")
        plt.savefig(path, dpi=300)
        plt.close()
        paths.append(path)
        
        
    else:
        idx = int(args_layer)
        if idx < 0:
            idx = total_layers + idx
        att = attns[idx][0, :, -1, pos:pos_end].mean(dim=0).cpu().numpy()
        general_att = general_attns[idx][0, :, -1, pos:pos_end].mean(dim=0).cpu().numpy()
        min_len = min(len(att), len(general_att))
        att = att[:min_len]
        general_att = general_att[:min_len]
        heat = att / (general_att + 1e-6)
        plt.figure(figsize=(5, 5))
        plt.imshow(heat.reshape(output_shape), cmap="viridis", interpolation="nearest")
        plt.axis("off")
        plt.tight_layout()
        path = os.path.join(save_dir, f"{img_base_name}_attn_layer{idx+1}.png")
        plt.savefig(path, dpi=300)
        plt.close()
        paths.append(path)
        
        npy_path = os.path.join(heatmap_save_dir, f"layer_{idx+1}.npy")
        np.save(npy_path, heat.reshape(output_shape))
    return paths



# ==== 坐标映射与拼接可视化（实心 + 加倍版）====
from PIL import Image, ImageDraw
from math import cos, sin, pi

def _first_pair_from_any(obj):
    if obj is None:
        return None
    if isinstance(obj, str):
        pairs = extract_coords(obj)
        if len(pairs) >= 1:
            return tuple(map(int, pairs[0]))
        m = re.search(r'(\d+)\s*[,;\s]\s*(\d+)', obj)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return None
    if isinstance(obj, (list, tuple)):
        if len(obj) == 2 and all(isinstance(v, (int, float)) for v in obj):
            return (int(obj[0]), int(obj[1]))
        if len(obj) >= 1:
            first = obj[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                return (int(first[0]), int(first[1]))
    return None

def _scale_1k_to_1080_2400(xy):
    if xy is None:
        return None
    x, y = xy
    return (int(round(x * 1080.0 / 1000.0)), int(round(y * 2400.0 / 1000.0)))

# ==== 三种实心绘制 ====

def _draw_circle(draw: ImageDraw.ImageDraw, center, r=28):
    """红色实心圆 (预测坐标)"""
    if center is None:
        return
    x, y = center
    draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 0, 0))

def _draw_triangle(draw: ImageDraw.ImageDraw, center, size=44):
    """蓝色实心等边三角形 (目标坐标)"""
    if center is None:
        return
    x, y = center
    p1 = (x, y - size)
    p2 = (x - size, y + size)
    p3 = (x + size, y + size)
    draw.polygon([p1, p2, p3], fill=(0, 102, 255))

def _draw_star(draw: ImageDraw.ImageDraw, center, r_out=40, r_in=16, n=5):
    """绿色实心五角星 (原图坐标)"""
    if center is None:
        return
    x0, y0 = center
    pts = []
    for k in range(2 * n):
        ang = -pi / 2 + k * pi / n
        r = r_out if k % 2 == 0 else r_in
        pts.append((x0 + int(r * cos(ang)), y0 + int(r * sin(ang))))
    draw.polygon(pts, fill=(0, 200, 0))

# ==== 拼接 & 主函数 ====

def _compose_side_by_side(left_img: Image.Image, right_img: Image.Image) -> Image.Image:
    tgt_w, tgt_h = 1080, 2400
    if left_img.size != (tgt_w, tgt_h):
        left_img = left_img.resize((tgt_w, tgt_h), Image.BILINEAR)
    if right_img.size != (tgt_w, tgt_h):
        right_img = right_img.resize((tgt_w, tgt_h), Image.BILINEAR)
    out = Image.new("RGB", (tgt_w * 2, tgt_h), (0, 0, 0))
    out.paste(left_img, (0, 0))
    out.paste(right_img, (tgt_w, 0))
    return out

def save_action_maps_for_results(
    results_json_path: str,
    imgs_root: str = "/data1/home/yanzihe/project/50samples/pop-up/imgs",
    verbose: bool = True
):
    """
    读取 <tier>/<variant>/<CFG_STR>/results.json，生成拼接可视化：
      左侧 原图 只画 raw_label（绿色五角星）。
      右侧 截图 同时画 预测 红色实心圆、目标 蓝色实心三角、原图 绿色实心五角星。
    保存路径：
      与 attn_maps 同级，写到 action_map/right 或 action_map/wrong。
    """
    if not os.path.isfile(results_json_path):
        raise FileNotFoundError(f"results.json not found: {results_json_path}")

    with open(results_json_path, "r", encoding="utf-8") as f:
        recs = json.load(f)

    base_dir = os.path.dirname(results_json_path)
    out_right = os.path.join(base_dir, "action_map", "right")
    out_wrong = os.path.join(base_dir, "action_map", "wrong")
    os.makedirs(out_right, exist_ok=True)
    os.makedirs(out_wrong, exist_ok=True)

    saved = []
    for rec in recs:
        try:
            img_path = rec.get("img_path", "")
            episode_id = str(rec.get("episodeID", "")).strip()
            step_id = int(rec.get("stepID", 0))
            is_close = bool(rec.get("is_close", False))
            base_name = os.path.splitext(os.path.basename(img_path))[0]

            raw_img_path = os.path.join(
                imgs_root, f"episode_{episode_id}_screenshot_{step_id}.png"
            )
            if not os.path.exists(raw_img_path):
                if verbose:
                    print(f"[warn] 原图缺失，跳过：{raw_img_path}")
                continue
            if not os.path.exists(img_path):
                if verbose:
                    print(f"[warn] 截图缺失，跳过：{img_path}")
                continue

            left = Image.open(raw_img_path).convert("RGB")
            right = Image.open(img_path).convert("RGB")

            raw_xy = _first_pair_from_any(rec.get("raw_label"))
            pred_xy_1k = _first_pair_from_any(rec.get("answer_1"))
            pred_xy = _scale_1k_to_1080_2400(pred_xy_1k) if pred_xy_1k else None
            gold_xy_1k = _first_pair_from_any(rec.get("gold_coords"))
            gold_xy = _scale_1k_to_1080_2400(gold_xy_1k) if gold_xy_1k else None

            l = left.copy()
            dl = ImageDraw.Draw(l)
            _draw_star(dl, raw_xy)

            r = right.copy()
            dr = ImageDraw.Draw(r)
            _draw_circle(dr, pred_xy)
            _draw_triangle(dr, gold_xy)
            _draw_star(dr, raw_xy)

            merged = _compose_side_by_side(l, r)
            out_dir = out_right if is_close else out_wrong
            save_path = os.path.join(out_dir, f"{base_name}_action_map.png")
            merged.save(save_path, quality=95)
            saved.append(save_path)
            if verbose:
                print(f"[ok] saved: {save_path}")
        except Exception as e:
            if verbose:
                print(f"[err] {rec.get('img_path','<unknown>')} -> {e}")
            continue
    return saved
