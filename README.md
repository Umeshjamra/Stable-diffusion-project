# 🎨 Comprehensive Text-to-Image Generation Pipeline

> **Internship Task** | Built on top of the Stable Diffusion training project — extended with GAN-based generation, text preprocessing, and text embedding creation.

---

## 📋 Problem Statement

Build a comprehensive text-to-image generating pipeline that includes:
1. **GAN-based image generation** (DCGAN architecture)
2. **Text preprocessing** (tokenization, stop-word removal, quality scoring, prompt enhancement)
3. **Text embedding creation** (vocabulary-based dense embeddings, cosine similarity, PCA visualization)
4. **Pipeline comparison** (GAN vs Stable Diffusion)

This project **extends the Stable Diffusion training project** with three new modules, all integrated into the same codebase.

---

## 🗂️ Project Structure

```
text-to-image-pipeline/
│
├── stable_diffusion_generator.py   # Original training project (SD 1.5 + Gradio UI)
├── text_to_image_pipeline.py       # Internship extension (GAN + Preprocessing + Embeddings)
├── requirements.txt
├── README.md
│
└── outputs/
    ├── embeddings/
    │   ├── token_distribution.png        # Token frequency & prompt quality scores
    │   └── embedding_visualization.png   # PCA + cosine similarity heatmap
    ├── gan_training/
    │   └── gan_training_metrics.png      # G/D loss curves, D(x), D(G(z))
    ├── generated/
    │   └── gan_text_to_image.png         # GAN-generated images per prompt
    └── comparison/
        ├── pipeline_architecture.png     # GAN vs SD pipeline diagram
        └── model_comparison.png          # Feature comparison table
```

---

## 🔧 Methodology

### Module 1 — Text Preprocessing (`TextPreprocessor`)

| Step | Method | Purpose |
|------|--------|---------|
| Cleaning | Regex normalization | Remove noise, normalize casing |
| Tokenization | Whitespace + regex | Split into meaningful tokens |
| Stop word removal | Custom set (36 words) | Retain descriptive content tokens |
| Quality scoring | Multi-dimensional (0–100) | Length (40pt) + quality keywords (30pt) + style (30pt) |
| Prompt enhancement | Keyword injection | Auto-append quality tokens if missing |

### Module 2 — Text Embedding Engine (`TextEmbeddingEngine`)

- Vocabulary built from corpus using `Counter`
- Dense embedding matrix initialized with `N(0, 0.01)` and L2-normalized (simulates CLIP)
- Token IDs looked up → mean-pooled → L2-normalized output vector
- Cosine similarity matrix for pairwise prompt comparison
- PCA (manual eigendecomposition) for 2D visualization

> In production: swap with `sentence-transformers` or HuggingFace CLIP for real semantic embeddings.

### Module 3 — GAN-Based Generation (`TextConditionedGAN` + DCGAN)

**Generator architecture:**
```
z [100] + text_emb [256] → ConvTranspose2d × 5 → 64×64 RGB
```

**Discriminator architecture:**
```
64×64 RGB → Conv2d × 5 → Sigmoid scalar
```

- Weight init: `N(0, 0.02)` (original DCGAN paper)
- Optimizer: Adam (lr=0.0002, β₁=0.5)
- Loss: Binary Cross Entropy
- Text conditioning: concatenate text embedding to latent noise z

### Module 4 — Pipeline Comparison

| Attribute | DCGAN (New) | Stable Diffusion (Training) |
|-----------|-------------|----------------------------|
| Architecture | G + D | U-Net + VAE + CLIP |
| Training | Adversarial | Denoising score matching |
| Inference speed | ~1 ms | ~5–30s |
| Image quality | Moderate (64×64) | High (512×512+) |
| Text conditioning | Embedding concat | Cross-attention |
| GPU memory | ~100 MB | ~4–8 GB |

---

## 📊 Results & Visualizations

### Text Preprocessing
![Token Distribution](outputs/embeddings/token_distribution.png)

### Embedding Visualization
![Embeddings](outputs/embeddings/embedding_visualization.png)

### GAN Training Metrics
![GAN Training](outputs/gan_training/gan_training_metrics.png)

### Generated Images (GAN)
![Generated](outputs/generated/gan_text_to_image.png)

### Pipeline Comparison
![Pipeline](outputs/comparison/pipeline_architecture.png)

---

## 🚀 Setup & Usage

### Install Dependencies
```bash
pip install torch torchvision diffusers transformers gradio numpy pillow matplotlib seaborn
```

### Run Training Project (Stable Diffusion UI)
```bash
python stable_diffusion_generator.py
# → Opens Gradio interface at http://localhost:7860
```

### Run Internship Extension (GAN + Preprocessing + Embeddings)
```bash
python text_to_image_pipeline.py
```

### Run Individual Modules
```python
from text_to_image_pipeline import TextPreprocessor, TextEmbeddingEngine, TextConditionedGAN

# Text Preprocessing
preprocessor = TextPreprocessor()
result = preprocessor.compute_prompt_score("a beautiful mountain at sunset")
print(result)  # {'total_score': 62.5, 'length_score': 24.0, ...}

# Text Embeddings
engine = TextEmbeddingEngine(embedding_dim=256)
engine.build_vocab(["your", "prompts", "here"])
embedding = engine.embed_text("a serene landscape")  # [256] tensor

# GAN Generation
gan = TextConditionedGAN(nz=100, embedding_dim=256)
images = gan.generate_from_text("cyberpunk city at night", n_images=4)
```

---

## 📦 Dataset

> **Google Drive Link**:

The dataset used includes:
- Demo corpus of 8 diverse text prompts for embedding/GAN training demonstration
- Synthetic image data (random tensors) for GAN training demo
- For production quality, replace with:
  - [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (faces)
  - [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) (general objects)
  - Custom domain dataset

---

## 📈 Feature Engineering

1. **Prompt quality score**: composite metric from token count, quality keyword presence, style words
2. **Content token ratio**: non-stop-word tokens / total tokens (higher = more descriptive)
3. **Embedding normalization**: L2-normalized unit vectors for cosine similarity
4. **Text-noise concatenation**: [z; text_emb] conditioning vector for the DCGAN Generator

---

## 🔬 Model Comparison

| Metric | Baseline (SD 1.5) | GAN Extension |
|--------|-------------------|---------------|
| Architecture | Latent Diffusion | DCGAN |
| Output size | 512×512 | 64×64 |
| Inference speed | ~15s / image | < 1ms |
| Text-image alignment | CLIP cross-attention | Embedding concat |
| Training stability | Stable (score matching) | Adversarial (unstable) |
| Reproducibility | Seed + scheduler | Seed + z |

---

## 📝 References

1. Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*
2. Radford et al. (2015). *Unsupervised Representation Learning with DCGANs*
3. Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*
4. Ho et al. (2020). *Denoising Diffusion Probabilistic Models*

---

## 👤 Author

**[UMESH JAMRA]**  
Project built: 2025
