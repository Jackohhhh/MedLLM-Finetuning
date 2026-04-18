# cad-finetune

基于 **Hugging Face Transformers**、**PEFT** 与 **DeepSpeed** 的 **CAD / 医疗文本二分类** 微调脚手架：支持 **QLoRA**（4bit + LoRA）、**LoRA**（bf16 全精度基座、无量化）、**全量微调**；以及加权损失、数据集过采样、命令行覆盖超参、离线评估与预测落盘。

---

## 功能概览

| 能力 | 说明 |
|------|------|
| **QLoRA** | [`scripts/train/finetune_qlora.sh`](scripts/train/finetune_qlora.sh)：4bit 基座 + LoRA；优化器默认 `paged_adamw_32bit`（bitsandbytes），显存占用小 |
| **LoRA** | [`scripts/train/finetune_lora.sh`](scripts/train/finetune_lora.sh)：无量化、bf16 基座 + LoRA；优化器默认 `adamw_torch`，显存明显高于 QLoRA |
| **全量微调** | [`scripts/train/finetune_full.sh`](scripts/train/finetune_full.sh)：无量化、无 LoRA |
| **DeepSpeed ZeRO** | `configs/deepspeed/zero2.json`（ZeRO-2）与 `zero3.json`（ZeRO-3）；`--deepspeed` 可切换，精度与 HF 对齐 |
| **加权 CE** | `WeightedTrainer` + 数据集 YAML 中的 `class_weights` |
| **评估** | `cli.eval`：加载 checkpoint，对测试集 `predict`，输出 `metrics.json` / `predictions.jsonl` |
| **统一配置** | 医疗任务共用 [`configs/experiments/medical.yaml`](configs/experiments/medical.yaml) + [`configs/models/seq_cls.yaml`](configs/models/seq_cls.yaml)；三个训练脚本仅通过 CLI 区分全量 / LoRA / QLoRA |

---

## 环境要求

- **Python ≥ 3.10**
- **Linux + NVIDIA GPU**（推荐；QLoRA / DeepSpeed 依赖 CUDA）
- **PyTorch** 需与显卡驱动、CUDA 版本匹配（由你本机安装）

---

## 安装

```bash
git clone https://github.com/Jackohhhh/ukb-cad-llm-finetuning.git
cd ukb-cad-llm-finetuning

# 可编辑安装（推荐开发）
pip install -e .

# 可选：FlashAttention-2（与 torch/CUDA 版本强相关）
pip install -e ".[flash]"
```

依赖声明见 [`pyproject.toml`](pyproject.toml)。


---

## 支持的模型

训练脚本中通过 **`--model-name-or-path`** 传入 Hugging Face Hub ID（或本地路径）。下表为常用示例；**Gated 模型**需在 [Hugging Face](https://huggingface.co/) 接受许可并配置 `HF_TOKEN`；部分医学 / 自定义权重需追加 **`--trust-remote-code true`**。BioBERT 等为**编码器**架构，与 7B/8B 因果语言模型在分类头与 LoRA 目标层上可能不同，换模型时请自行核对显存与兼容性。

### 约 1B（暂未测试）

| 名称 | Hub ID |
|------|--------|
| BioBERT | `dmis-lab/biobert-v1.1` |

### 医用 / 医学向（7B～8B）

| 名称 | Hub ID |
|------|--------|
| MedAlpaca | `medalpaca/medalpaca-7b` |
| Meditron | `epfl-llm/meditron-7b` |
| Med42 | `m42-health/Llama3-Med42-8B` |

### 通用大模型（7B～8B）

| 名称 | Hub ID |
|------|--------|
| Qwen2 | `Qwen/Qwen2-7B-Instruct` |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` |
| Gemma 2 | `google/gemma-2-9b-it` |
| Llama 2 | `meta-llama/Llama-2-7b-chat-hf` |
| Llama 3 | `meta-llama/Meta-Llama-3-8B-Instruct` |


---

## 数据准备

默认数据集配置见 [`configs/datasets/medical_binary.yaml`](configs/datasets/medical_binary.yaml)：

- **训练集路径**：`data/raw/medical_train.json`
- **测试集路径**：`data/raw/medical_test.json`
- **验证集路径**：未指定独立验证集时，从训练集按 **`validation_split`** 划分验证集

JSON 字段由 [`configs/tasks/binary_classification.yaml`](configs/tasks/binary_classification.yaml) 定义（默认文本列 **`input`**，标签列 **`output`**，值为 `0` / `1`）。

---

## 快速开始

在项目根目录执行（脚本内已 `cd` 到仓库根并设置 `PYTHONPATH`）。
**多卡**：在所用脚本中把 `deepspeed --num_gpus=1` 改成`deepspeed --num_gpus=N`。

### QLoRA（4bit 量化）

脚本：[`scripts/train/finetune_qlora.sh`](scripts/train/finetune_qlora.sh)。默认 `--load-in-4bit true`、`--bf16`、`--optim paged_adamw_32bit`。检查点目录示例：`outputs/checkpoints/Qwen_Qwen2-7B-Instruct_qlora/`（随 `--model-name-or-path` 变化）。

```bash
bash scripts/train/finetune_qlora.sh
```

### LoRA（bf16 全精度基座）

脚本：[`scripts/train/finetune_lora.sh`](scripts/train/finetune_lora.sh)。默认 `--load-in-4bit false`、`--bf16`、`--optim adamw_torch`；基座以 bf16 加载，**显存显著高于 QLoRA**。检查点目录示例：`outputs/checkpoints/Qwen_Qwen2-7B-Instruct_lora/`。

```bash
bash scripts/train/finetune_lora.sh
```


### 全量微调

脚本：[`scripts/train/finetune_full.sh`](scripts/train/finetune_full.sh)。

```bash
bash scripts/train/finetune_full.sh
```

### 评估测试

```bash
bash scripts/eval/eval_binary_cls.sh --checkpoint outputs/checkpoints/..
```

### 仅调用 Python 模块

```bash
export PYTHONPATH=src

python -m cad_finetune.cli.train --config configs/experiments/medical.yaml ...
python -m cad_finetune.cli.eval --config configs/experiments/medical.yaml --checkpoint <路径> ...
```

---

## 配置说明

| 路径 | 作用 |
|------|------|
| `configs/experiments/medical.yaml` | 医疗二分类统一实验入口：`paths` 引用 `seq_cls.yaml`、task、dataset、deepspeed；训练时 |
| `configs/models/seq_cls.yaml` | 序列分类骨架：默认 `model_name_or_path`、LoRA 结构等；可被 CLI `--model-name-or-path` 覆盖 |
| `configs/datasets/*.yaml` | 数据路径、划分数据集、正样本过采样、`class_weights` 权重 |
| `configs/tasks/*.yaml` | 列名、`max_length` 最大长度、类别数 |
| `configs/deepspeed/*.json` | DeepSpeed加速 |




---

## 输出文件

| 类型 | 位置 |
|------|----------------------------|
| 检查点 / 适配器 | `outputs/checkpoints/<model_slug>_<模式>/`（模式为 full、lora、qlora 或 lora8bit） |
| 训练后测试集预测 | `outputs/predictions/<同上>/` 下的 `metrics.json`、`predictions.jsonl` |
| 训练曲线 / 指标 | `--report-to wandb` 时见 wandb 网页与项目下 `wandb/`；需先执行 `wandb login` |

---

## 常见问题

### 环境与依赖

**1.模型从哪里下载？**  
`model_name_or_path` 为 Hub ID（如 `Qwen/Qwen2-7B-Instruct`）时，首次运行会缓存到本机 Hugging Face 默认目录（或你设置的 `HF_HOME` / `cache_dir`）。

**2.未设置 `HF_TOKEN` 时 Hub 很慢或限流？**  
公开模型仍可下载；若频繁 429 或很慢，建议在 [Hugging Face](https://huggingface.co/settings/tokens) 创建 Token，并在环境中设置 **`HF_TOKEN`**（或 `huggingface-cli login`）。


**3.Transformers 升级大版本后参数报错？**  
本仓库已对 **`eval_strategy` / `processing_class`** 等与 v5 差异做兼容；若仍遇 `TrainingArguments` / `Trainer` 参数变化，请对照当前安装的 `transformers` 发行说明，或把 `pyproject.toml` 中的版本范围锁在已知可用区间。

### 训练

**1.显存不足（OOM）？**  
优先：减小 **`--per-device-train-batch-size`**、增大 **`--gradient-accumulation-steps`**；或使用 **QLoRA**、多卡 **ZeRO-3**、开启 **`--gradient-checkpointing`**。


### 检查点与评估路径

**1.`load_best_model_at_end` 保存的是「最后一轮的参数权重」吗？**  
**不一定。** 开启后保存的是**验证集上 `metric_for_best_model` 最优**（如 F1）对应的权重。若要用某一固定 step，请指向 **`checkpoint-<step>`**。

- **`--checkpoint`**：必须指向 **`outputs/checkpoints/...`** 下的模型或 **`checkpoint-*`** 子目录（含适配器或全量权重）。  


---

## 更新日志

| 日期 | 说明 |
|------|------|
| 2026-04-17 | 基于 DeepSpeed 完成 Qwen2 的 LoRA、QLoRA 与全量微调的工程实现。 |

---

## 许可证
[MIT](https://github.com/Jackohhhh/ukb-cad-llm-finetuning/blob/main/LICENSE)

---

## 致谢

构建于 [Transformers](https://github.com/huggingface/transformers)、[PEFT](https://github.com/huggingface/peft)、[DeepSpeed](https://github.com/microsoft/DeepSpeed) 等开源项目之上。
