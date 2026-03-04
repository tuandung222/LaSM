# Readme

This supplementary repository contains all necessary code and datasets to fully reproduce the experiments presented in our dissertation. It includes two main folders: `Source` and `Dataset`, as well as a `envirnment.yml` file to facilitate Conda environment setup.

## Code

We provide scripts for the Qwen series models as examples.

### Qwen Series (Qwen2-VL-2/7/70B, OS-Atlas)

- Use the script below to compute:

  - **DSR (Defense Success Rate)** — the key evaluation metric used in our work.
  - **Heatmaps** of attention for all layers or a specific layer, along with the corresponding heat score data (`.npy` files).
  - **Model outputs** for user queries, including both:
    - the answer to the user's goal, and
    - a general description of the screenshot.

```shell
python lams_qwen.py \
  --layer all \
  --model_path /your/model/path \
  --input_dir /path/to/images \
  --output_dir /path/to/save/results \
  --lower_bound 7 \
  --upper_bound 18 \
  --alpha 1.1
```

#### Output Folder Structure

Running the script will generate the following directory structure:

```bash
results
└── 7_17_1dot1
    ├── attn_maps
    │   ├── right
    │   │   ├── heatmap_values
    │   │   │   ├── img1
    │   │   │   │   ├── layer_1.npy
    │   │   │   │   ├── layer_2.npy
    │   │   │   ├── img2
    │   │   │   │   ├── layer_1.npy
    │   │   │   │   ├── layer_2.npy
    │   │   ├── img1_attn_all_layers.png
    │   │   ├── img2_attn_all_layers.png
    │   ├── wrong
    │   │   ├── heatmap_values
    │   │   │   ├── img1
    │   │   │   │   ├── layer_1.npy
    │   │   │   ├── img2
    │   │   │   │   ├── layer_1.npy
    │   │   ├── img1_attn_all_layers.png
    │   │   ├── img2_attn_all_layers.png
    ├── results.json
    ├── meta.json

```

### LLaMA and LLaVA Series

Use the appropriate script for LLaMA or LLaVA models:

```bash
# Choose lams_llama.py or lams_llava.py depending on the model
python lams_llama.py \
  --model_path /your/model/path \
  --input_dir /path/to/images \
  --output_dir /path/to/save/results \
  --lower_bound 12 \
  --upper_bound 28 \
  --alpha 1.1
```

# Dataset 

The examples of pop-ups in different style are illustrated in `Dataset`  folder.

