# VLLM_HuggingFace_Perf_Benchmark

# LLM Inference Benchmark: vLLM inference engine + INT4 Quantization vs Hugging Face (Facebook/OPT-125M)

[![Open in Colab](https://img.shields.io/badge/Colab-Open-orange)](https://colab.research.google.com/github/<your-username>/llm-inference-bench-vllm-int4-vs-huggingface/blob/main/notebooks/colab_benchmark.ipynb)
![GPU](https://img.shields.io/badge/GPU-T4-blue)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

Run **bigger LLMs on smaller devices** with:
- **vLLM** (PagedAttention) inference engine
- **INT4 quantization** to reduce model memory size

This repo reproduces a **real-world benchmark** comparing **Hugging Face FP16 inference** vs **vLLM with INT4 quantized inference** on **Facebook/OPT-125M**.

> TL;DR (from our Colab T4 run):
- **Latency:** 1.02s → **0.40s** (↓ ~61%)
- **Throughput:** 61.38 → **176.70 tokens/s** (~**2.9×**)
- **Model memory size:** **−75%** (INT4 vs FP16)

![Hero](assets/hero/hero_card_no_overlap.png)

## 🔬 Process Difference

![vLLM vs Hugging Face process](assets/hero/process_vllm_vs_hf_5items_clean.png)

> vLLM combines **PagedAttention**, **INT4 quantized weights (BitsAndBytes)**, **continuous batching**, and **optimized CUDA kernels**, which drives higher throughput, lower latency, and **~75% reduced model memory size** vs the standard Hugging Face FP16 baseline.


## ✨ Results

## Improvements
![vLLM vs Hugging Face process](assets/charts/table.png)

## 75% reduced Model Size
![vLLM vs Hugging Face process](assets/charts/04_model_size_v3_fixed.png)

## Average Throughput
![vLLM vs Hugging Face process](assets/charts/03_throughput_v3_fixed.png)

## Average Latency
![vLLM vs Hugging Face process](assets/charts/01_latency_avg_v3_fixed.png)

## Tail Latency
![vLLM vs Hugging Face process](assets/charts/02_latency_tail_v3_fixed.png)

## Whats Inside
- One-click **Colab notebook** for T4 GPU
- **Beautiful charts + graphs** for visualization of benchmarking
- Clear **methodology** + **troubleshooting** notes

---

## 🚀 Quickstart

### Google Colab
1. Open the notebook: **[vllm_quant_inference_benchmark.ipynb](https://colab.research.google.com/github/ashishmokalkar/VLLM_HuggingFace_Perf_Benchmark/blob/main/vllm_quant_inference_benchmark.ipynb)**  in google colab
2. `Runtime → Change runtime type → GPU (T4)`
3. Run all cells (installs deps, runs benchmark, saves the results in csv and json file ).
