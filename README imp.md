# 🎨 Advanced Text-to-Image Generation Pipeline
### Internship Project — All 6 Tasks | Built on Stable Diffusion

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Umeshjamra/Stable-diffusion-project/blob/main/advanced_text_to_image_all_tasks.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.21.0-orange.svg)](https://huggingface.co/docs/diffusers)
[![Gradio](https://img.shields.io/badge/Gradio-4.0-green.svg)](https://gradio.app)

> ⚠️ **Run on Google Colab with GPU** — Runtime → Change runtime type → **T4 GPU**

---

## 📋 Problem Statement

Extend a Stable Diffusion training project into a **comprehensive text-to-image pipeline** that integrates:
1. Real CLIP + BERT text embeddings via HuggingFace Transformers
2. Self-Attention (SAGAN) and Cross-Attention inside a GAN
3. Fine-tuning techniques: LoRA and Textual Inversion
4. Public dataset exploration and analysis (Oxford 102 Flowers)
5. Multi-model text preprocessing pipeline (CLIP + BERT)
6. Conditional GAN (CGAN) generating shapes from labels

All modules are integrated into a **single Gradio UI** with 5 interactive tabs.

---

## 🗂️ Repository Structure

```
Stable-diffusion-project/
│
├── advanced_text_to_image_all_tasks.ipynb   # 🔑 MAIN — all 6 tasks in one notebook
│
├── dataset/
│   ├── oxford102_flowers_dataset.csv        # Task 4 — all 102 flower classes + stats
│   ├── cgan_shape_dataset.csv               # Task 6 — 2000 shape samples (4 classes × 500)
│   ├── text_embeddings_dataset.csv          # Task 5 — 12 prompts, token counts, embeddings
│   ├── cgan_training_log.csv                # Tasks 1,2,6 — 40-epoch CGAN training history
│   ├── sd_generation_log.csv                # Tasks 1,3 — SD generation metadata (10 runs)
│   ├── model_comparison.csv                 # All tasks — baseline vs advanced benchmark
│   └── dataset_info.json                    # Metadata: description, row counts, task mapping
│
├── outputs/                                 # Auto-created when notebook runs
│   ├── task2_attention_maps.png
│   ├── task4_dataset_analysis.png
│   ├── task4_samples.png
│   ├── task5_text_embeddings.png
│   ├── task6_cgan_results.png
│   └── sd_*.png  (generated images)
│
└── README.md
```

---

## 📌 Tasks Overview

| # | Task | Module | Key Output |
|---|------|--------|-----------|
| 1 | Comprehensive Pipeline | `ComprehensiveTTIPipeline` | End-to-end: text → preprocess → CLIP embed → CGAN/SD |
| 2 | Attention Mechanisms | `SelfAttention`, `CrossAttention` | `task2_attention_maps.png` |
| 3 | Fine-Tuning Stable Diffusion | `LoRALinear`, `TextualInversionEmbedding` | Domain-adapted SD |
| 4 | Dataset Exploration | `DatasetAnalyser` | `task4_dataset_analysis.png` |
| 5 | Text Preprocessing & Embeddings | `TextEmbeddingPipeline` | `task5_text_embeddings.png` |
| 6 | Conditional GAN | `CGANGenerator`, `CGANDiscriminator` | `task6_cgan_results.png` |

---

## 🧠 Architecture

### Overall System

```
Prompt Text
    │
    ▼
[Task 5] TextEmbeddingPipeline
    ├── CLIP (openai/clip-vit-base-patch32) → (batch, seq_len=77, dim=512)
    └── BERT (bert-base-uncased) → (batch, dim=768) [CLS token]
    │
    ▼
[Task 1] ComprehensiveTTIPipeline
    ├── CGAN Path ──► [Task 6] CGANGenerator → 32×32 shape image (fast)
    └── SD Path   ──► StableDiffusionPipeline → 512×512 high-quality image
```

### Task 2 — Attention Mechanisms

**Self-Attention (SAGAN-style):**
```python
# q: (B, HW, C/8)   k: (B, C/8, HW)   v: (B, C, HW)
attn = softmax(q @ k)          # (B, HW, HW) — every position attends to all others
out  = gamma * (v @ attn^T) + x  # learned residual mix
```

**Cross-Attention (text → image):**
```python
# img_feat: (B, C, H, W)    text_emb: (B, T, text_dim)
q = Wq(img_feat)   # image queries
k = Wk(text_emb)   # text keys
v = Wv(text_emb)   # text values
attn = softmax(q @ k^T / sqrt(d_head))
out  = attn @ v + img_feat   # residual
```

### Task 6 — CGAN Architecture

```
Generator G(z, label):
  z [100] + label_emb(label) [16] → concat [116]
  → Linear [116 → 256×8×8]
  → ConvTranspose2d: 8→16 (128ch) → 16→32 (64ch)
  → SelfAttention(64)               ← Task 2 integration
  → Conv2d → Tanh  →  [1, 32, 32]

Discriminator D(img, label):
  label_emb(label) → project to [1, 32, 32] spatial map
  concat [img:1ch, label_map:1ch] → [2, 32, 32]
  → Conv2d ×2 → Flatten → Linear → Sigmoid → P(real|label)
```

---

## 📊 Dataset

**Google Drive:** [📁 Dataset Folder](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)

| File | Rows | Task | Description |
|------|------|------|-------------|
| `oxford102_flowers_dataset.csv` | 102 | Task 4 | All 102 Oxford Flowers classes: sample count, caption length, resolution, dominant color |
| `cgan_shape_dataset.csv` | 2,000 | Task 6 | ShapeDataset: 500 samples per class (circle/square/triangle/star), geometry parameters |
| `text_embeddings_dataset.csv` | 12 | Task 5 | 12 prompts with CLIP token counts, word stats, 16-dim embedding vectors |
| `cgan_training_log.csv` | 40 | T1,2,6 | CGAN training: D_loss, G_loss, D(x), D(G(z)) per epoch × 40 epochs |
| `sd_generation_log.csv` | 10 | T1,3 | SD generation metadata: model, scheduler, steps, CFG scale, seed, timing |
| `model_comparison.csv` | 6 | All | Baseline→CGAN→Attention→SD→LoRA→Textual Inversion benchmark |

**Total: 2,170 rows across 6 files**

---

## 📈 Results & Visualisations

### Task 2 — Self-Attention and Cross-Attention Maps
![Task 2](outputs/task2_attention_maps.png)
*Left: Input feature map | Centre: Self-attention from centre pixel | Right: After cross-attention with text*

### Task 4 — Oxford 102 Flowers Dataset Analysis
![Task 4](outputs/task4_dataset_analysis.png)
*Top-20 class distribution · Dataset summary table · Caption length histogram · Resolution scatter · Top-8 pie chart*

### Task 5 — CLIP Tokenisation and Embedding Similarity
![Task 5](outputs/task5_text_embeddings.png)
*CLIP token count per prompt (vs 77 max) + cosine similarity heatmap of prompt embeddings*

### Task 6 — CGAN Shape Generation
![Task 6](outputs/task6_cgan_results.png)
*Training loss curves · Generated shapes grid (circle, square, triangle, star) · One clear sample per class*

---

## 🔬 Model Comparison

| Model | Task | Arch | Output | Inference | VRAM | Text Align | Quality |
|-------|------|------|--------|-----------|------|-----------|---------|
| DCGAN (Baseline) | T1 | GAN | 32×32 | ~1ms | 50MB | Low | ★★★☆☆ |
| CGAN | T6 | Conditional GAN | 32×32 | ~1ms | 80MB | High (class) | ★★★☆☆ |
| CGAN + Self+Cross Attn | T2+T6 | CGAN+Attn | 32×32 | ~2ms | 100MB | High | ★★★★☆ |
| SD 1.5 (training base) | T1 | LDM+CLIP | 512×512 | ~12s | 4.5GB | High | ★★★★★ |
| SD 1.5 + LoRA | T3 | LDM+LoRA | 512×512 | ~13s | 4.8GB | High | ★★★★★ |
| SD 1.5 + Textual Inv. | T3 | LDM+TI | 512×512 | ~12s | 4.5GB | Very High | ★★★★★ |

**Key insight:** The CGAN + Self-Attention model (Task 2+6) is ~6000× faster than SD with ~45× less VRAM, at the cost of output resolution and realism. Fine-tuning SD with LoRA (Task 3) provides domain adaptation with only ~0.1% extra trainable parameters.

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/Umeshjamra/Stable-diffusion-project.git
cd Stable-diffusion-project
```

**Option A — Colab (recommended):**
1. Open [Google Colab](https://colab.research.google.com)
2. File → Open notebook → GitHub → `Umeshjamra/Stable-diffusion-project`
3. Select `advanced_text_to_image_all_tasks.ipynb`
4. Runtime → Change runtime type → **T4 GPU**
5. Runtime → Run all

**Option B — Local:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.21.0 transformers==4.30.2 accelerate==0.20.3
pip install xformers gradio==4.0.0 datasets pillow matplotlib seaborn scipy
jupyter notebook advanced_text_to_image_all_tasks.ipynb
```

---

## 🖥️ Gradio UI (5 Tabs)

| Tab | Task | Description |
|-----|------|-------------|
| 🚀 Stable Diffusion | T1 | Full SD generation with scheduler, seed, size, CFG controls |
| 🔷 CGAN Shapes | T6 | Generate circle/square/triangle/star from text prompt |
| 📝 Text Embeddings | T5 | Paste prompts → get word stats + CLIP analysis |
| 🖼️ Examples & Gallery | T1 | Example prompts + recent generation gallery |
| 📚 About the Project | All | Architecture table, key technologies |

---

## 📦 Dependencies

```
torch>=2.0         torchvision       diffusers==0.21.0
transformers==4.30.2   accelerate==0.20.3   xformers==0.0.20
gradio==4.0.0      datasets           Pillow==9.5.0
numpy==1.24.4      matplotlib==3.7.2  seaborn   scipy
```

---

## 📚 References

1. Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models.* CVPR.
2. Zhang et al. (2018). *Self-Attention Generative Adversarial Networks.* ICML 2019.
3. Gal et al. (2022). *An Image is Worth One Word: Textual Inversion.* ICLR 2023.
4. Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022.
5. Mirza & Osindero (2014). *Conditional Generative Adversarial Nets.* arXiv.
6. Radford et al. (2021). *Learning Transferable Visual Models From Natural Language (CLIP).* ICML.
7. Devlin et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL 2019.
8. Nilsback & Zisserman (2008). *Automated Flower Classification over a Large Number of Classes.* ICVGIP.

---

## 👤 Author

**Umesh Jamra** | Internship 2025
**GitHub:** [github.com/Umeshjamra/Stable-diffusion-project](https://github.com/Umeshjamra/Stable-diffusion-project)
