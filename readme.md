<h1 align="center" style="font-size: 25px;">LycheeMemory: Dynamic Long Context Reasoning over Compressed Memory via End-to-End Reinforcement Learning</h1>

---

<p align="center">
  <a href="https://arxiv.org/abs/2602.08382">
    <img src="https://img.shields.io/badge/%F0%9F%93%84_PAPER-PDF-8A2BE2?style=for-the-badge" alt="Paper">
  </a>
  <a href="https://huggingface.co/lerverson/LycheeMemory-7B">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97_MODEL_WEIGHTS-HF-87CEEB?style=for-the-badge" alt="Model Weights">
  </a>
</p>

> **Note:** Currently, we have released the **inference code w/o Gate and model weights**. The training pipeline (including Compressor pre-training, RL optimization, and Gate training) will be released soon.

LycheeMemory is a cognitively inspired framework that enables efficient long-context inference via **chunk-wise compression** and **selective memory recall**. By mimicking the division of labor between human long-term memory and dynamic working memory, LycheeMemory achieves multi-hop reasoning over massive contexts (extrapolating up to **1.75M tokens**) while dramatically reducing computational overhead.

---

## 🌟 Key Features

* **Massive Context Scaling:** Successfully extrapolates context length from 7K to **1.75M tokens** with minimal performance degradation.
* **Efficiency:** In long-text processing，Compared to strong baselines like MemAgent, LycheeMemory achieves an average **2× reduction in peak GPU memory** usage and a **6× speedup** during inference, exhibiting near-constant inference latency as context grows.
* **Smart Dynamic Recall:** Uses a lightweight LoRA `Gate` to adaptively retrieve relevant memory chunks based on the *evolving working memory*, avoiding premature anchoring and the context fragmentation issues common in standard RAG.

## 🏗️ Architecture

<p align="center">
  <img src="figs/LycheeMemory.svg" width="700" alt="LycheeMemory Architecture">
</p>

LycheeMemory operates in two main phases:
1. **Memory Compression:** The `Compressor` segments long input documents into chunks (e.g., 4096 tokens) and compresses them into high-fidelity, compact KV-cache representations (Static Memory).
2. **Dynamic Recall and Reasoning:** Driven by a `Gate` and a `Reasoner`, the model starts with an empty working memory. It linearly scans the compressed memory bank, using the `Gate` to filter out irrelevant chunks. Relevant chunks trigger the `Reasoner` to iteratively update the working memory to synthesize the final answer.

## 🚀 Quick Start (Inference)

### Prerequisites

Ensure you have installed the required dependencies. Flash Attention 2 is highly recommended for optimal performance.

```bash
conda create -n lychee python=3.10
conda activate lychee

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

For long contexts, it is recommended to load the text and query from a JSON file. The file should contain context and optionally an extra_info.question field.
```Bash
python example.py \
    --model_path lerverson/LycheeMemory-7B \
    --text_file test.json \
    --row_idx 0 \
    --max_new_tokens 1024
```

If you find LycheeMemory useful in your research, please consider citing our paper:
```bash
@article{chen2026lycheememory,
  title={Dynamic Long Context Reasoning over Compressed Memory via End-to-End Reinforcement Learning},
  author={Chen, Zhuoen and Li, Dongfang and Zhang, Meishan and Hu, Baotian and Zhang, Min},
  journal={arXiv preprint arXiv:2602.08382},
  year={2026},
}
```