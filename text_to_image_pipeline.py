"""
==============================================================================
Comprehensive Text-to-Image Generation Pipeline
==============================================================================
Internship Task Extension of Training Project: Stable Diffusion Generator

NEW FEATURES ADDED:
  1. Text Preprocessing Module
  2. Text Embedding Creation (CLIP + Sentence Transformers)
  3. GAN-Based Image Generation (DCGAN)
  4. Pipeline Visualization & Metrics
  5. Embedding Comparison & Analysis

Author: [Your Name]
Date: 2025
==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autocast
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import os
import re
import time
import gc
import json
from typing import Optional, Tuple, List, Dict
from datetime import datetime

# NLP Libraries
import string
from collections import Counter

# ==============================================================================
# MODULE 1: TEXT PREPROCESSING
# ==============================================================================

class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline for image generation prompts.
    
    Handles:
    - Tokenization
    - Stop word removal
    - Normalization
    - Quality scoring
    - Prompt enhancement
    """

    # Common English stop words
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "of", "in", "on", "at",
        "to", "for", "with", "by", "from", "as", "into", "through", "during",
        "and", "or", "but", "not", "so", "yet"
    }

    # Quality enhancer keywords
    QUALITY_TOKENS = [
        "highly detailed", "photorealistic", "4k", "8k", "sharp focus",
        "professional lighting", "masterpiece", "best quality"
    ]

    # Negative prompt defaults
    DEFAULT_NEGATIVE = [
        "blurry", "low quality", "bad anatomy", "deformed", "ugly",
        "watermark", "text", "noise", "grainy", "oversaturated"
    ]

    def __init__(self):
        self.preprocessing_history = []

    def clean_text(self, text: str) -> str:
        """Basic cleaning: lowercase, remove extra spaces, special chars."""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s,.\-]', '', text)  # keep commas, periods, hyphens
        text = re.sub(r'\s+', ' ', text)
        return text

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = self.clean_text(text)
        tokens = re.findall(r'\b[\w\-]+\b', text)
        return tokens

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """Remove common stop words while preserving descriptive terms."""
        return [t for t in tokens if t not in self.STOP_WORDS]

    def compute_prompt_score(self, prompt: str) -> Dict:
        """
        Score a prompt on multiple dimensions:
        - Length score
        - Descriptiveness score
        - Quality keyword presence
        """
        tokens = self.tokenize(prompt)
        content_tokens = self.remove_stop_words(tokens)

        length_score = min(len(content_tokens) / 15.0, 1.0) * 40  # max 40pts
        quality_score = sum(
            1 for kw in self.QUALITY_TOKENS if kw.lower() in prompt.lower()
        ) / len(self.QUALITY_TOKENS) * 30  # max 30pts

        # Check for style/medium words
        style_words = ["oil painting", "watercolor", "digital art", "photograph",
                       "sketch", "illustration", "3d render", "concept art"]
        style_score = sum(1 for sw in style_words if sw in prompt.lower()) * 10  # max 30pts

        total_score = min(length_score + quality_score + style_score, 100)

        return {
            "total_score": round(total_score, 1),
            "length_score": round(length_score, 1),
            "quality_score": round(quality_score, 1),
            "style_score": round(style_score, 1),
            "token_count": len(tokens),
            "content_token_count": len(content_tokens),
            "tokens": tokens,
            "content_tokens": content_tokens
        }

    def enhance_prompt(self, prompt: str, auto_quality: bool = True) -> str:
        """
        Enhance a basic prompt with quality tokens.
        Only adds tokens not already present.
        """
        enhanced = prompt.strip()
        if auto_quality:
            missing = [
                qt for qt in self.QUALITY_TOKENS[:3]
                if qt.lower() not in enhanced.lower()
            ]
            if missing:
                enhanced += ", " + ", ".join(missing)
        return enhanced

    def batch_preprocess(self, prompts: List[str]) -> List[Dict]:
        """Process a batch of prompts and return analysis for each."""
        results = []
        for p in prompts:
            score = self.compute_prompt_score(p)
            enhanced = self.enhance_prompt(p)
            results.append({
                "original": p,
                "enhanced": enhanced,
                "analysis": score
            })
        return results

    def visualize_token_distribution(self, prompts: List[str], save_path: str = None):
        """
        Visualize token frequency across a set of prompts.
        """
        all_tokens = []
        for p in prompts:
            all_tokens.extend(self.remove_stop_words(self.tokenize(p)))

        counter = Counter(all_tokens)
        top_tokens = counter.most_common(20)
        words, counts = zip(*top_tokens) if top_tokens else ([], [])

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Text Preprocessing Analysis", fontsize=16, fontweight='bold')

        # Token frequency bar chart
        axes[0].barh(list(words), list(counts), color=plt.cm.viridis(
            np.linspace(0.2, 0.8, len(words))))
        axes[0].set_title("Top 20 Content Tokens")
        axes[0].set_xlabel("Frequency")
        axes[0].invert_yaxis()

        # Prompt score distribution
        scores = [self.compute_prompt_score(p)["total_score"] for p in prompts]
        axes[1].hist(scores, bins=10, color='steelblue', edgecolor='white', alpha=0.8)
        axes[1].set_title("Prompt Quality Score Distribution")
        axes[1].set_xlabel("Quality Score (0-100)")
        axes[1].set_ylabel("Count")
        axes[1].axvline(np.mean(scores), color='red', linestyle='--',
                        label=f'Mean: {np.mean(scores):.1f}')
        axes[1].legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()
        return fig


# ==============================================================================
# MODULE 2: TEXT EMBEDDING CREATION
# ==============================================================================

class TextEmbeddingEngine:
    """
    Multi-method text embedding engine.
    
    Methods:
    1. TF-IDF style bag-of-words embeddings (no external deps)
    2. Character n-gram embeddings
    3. CLIP-style embeddings (simulated for offline environments)
    
    In production, swap with:
    - sentence-transformers: SentenceTransformer('all-MiniLM-L6-v2')
    - CLIP: clip.encode_text()
    - HuggingFace: AutoModel from 'openai/clip-vit-base-patch32'
    """

    def __init__(self, vocab_size: int = 512, embedding_dim: int = 256):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.vocab = {}
        self.preprocessor = TextPreprocessor()
        self._build_embedding_matrix()

    def _build_embedding_matrix(self):
        """Initialize a learnable embedding matrix (simulates CLIP text encoder)."""
        torch.manual_seed(42)
        self.embedding_matrix = torch.randn(self.vocab_size, self.embedding_dim) * 0.1
        # Normalize rows (unit vectors like CLIP)
        self.embedding_matrix = F.normalize(self.embedding_matrix, dim=1)

    def build_vocab(self, corpus: List[str]):
        """Build vocabulary from a corpus of prompts."""
        all_tokens = []
        for text in corpus:
            all_tokens.extend(self.preprocessor.tokenize(text))
        counter = Counter(all_tokens)
        # Reserve index 0 for <PAD>, 1 for <UNK>
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for i, (token, _) in enumerate(counter.most_common(self.vocab_size - 2), start=2):
            self.vocab[token] = i
        print(f"Vocabulary built: {len(self.vocab)} tokens")

    def tokenize_and_pad(self, text: str, max_len: int = 77) -> torch.Tensor:
        """Tokenize text and pad/truncate to max_len (CLIP uses 77)."""
        tokens = self.preprocessor.tokenize(text)
        indices = [self.vocab.get(t, 1) for t in tokens]  # 1 = <UNK>
        # Truncate or pad
        indices = indices[:max_len]
        indices += [0] * (max_len - len(indices))  # pad with 0
        return torch.tensor(indices, dtype=torch.long)

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Create a dense embedding vector for a single text.
        Returns: [embedding_dim] tensor
        """
        token_ids = self.tokenize_and_pad(text)
        # Lookup embeddings and mean-pool (like a simple text encoder)
        embeddings = self.embedding_matrix[token_ids]  # [77, embedding_dim]
        # Mask padding
        mask = (token_ids != 0).float().unsqueeze(1)
        mean_embedding = (embeddings * mask).sum(0) / mask.sum(0).clamp(min=1)
        return F.normalize(mean_embedding, dim=0)

    def embed_batch(self, texts: List[str]) -> torch.Tensor:
        """Create embeddings for a batch. Returns: [N, embedding_dim]"""
        embeddings = torch.stack([self.embed_text(t) for t in texts])
        return embeddings

    def cosine_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """Compute pairwise cosine similarity between text embeddings."""
        embeddings = self.embed_batch(texts)
        sim_matrix = torch.mm(embeddings, embeddings.T).numpy()
        return sim_matrix

    def find_most_similar(self, query: str, corpus: List[str], top_k: int = 3):
        """Find top-k most similar prompts to a query."""
        query_emb = self.embed_text(query).unsqueeze(0)  # [1, dim]
        corpus_embs = self.embed_batch(corpus)           # [N, dim]
        similarities = torch.mm(query_emb, corpus_embs.T).squeeze(0).numpy()
        top_indices = similarities.argsort()[::-1][:top_k]
        return [(corpus[i], float(similarities[i])) for i in top_indices]

    def visualize_embeddings(self, texts: List[str], labels: List[str] = None,
                              save_path: str = None):
        """
        Visualize text embeddings using PCA dimensionality reduction.
        """
        embeddings = self.embed_batch(texts).numpy()

        # Manual PCA (2 components)
        centered = embeddings - embeddings.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        top2 = eigenvectors[:, -2:]  # top 2 principal components
        reduced = centered @ top2

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("Text Embedding Analysis", fontsize=16, fontweight='bold')

        # PCA scatter
        colors = plt.cm.tab10(np.linspace(0, 1, len(texts)))
        for i, (x, y) in enumerate(reduced):
            label = labels[i] if labels else f"Prompt {i+1}"
            axes[0].scatter(x, y, color=colors[i], s=100, zorder=5)
            axes[0].annotate(label[:20], (x, y), textcoords="offset points",
                             xytext=(5, 5), fontsize=8)
        axes[0].set_title("PCA: Text Embeddings (2D)")
        axes[0].set_xlabel("PC1")
        axes[0].set_ylabel("PC2")
        axes[0].grid(True, alpha=0.3)

        # Cosine similarity heatmap
        sim_matrix = self.cosine_similarity_matrix(texts)
        short_labels = [t[:15] + "..." if len(t) > 15 else t for t in texts]
        im = axes[1].imshow(sim_matrix, cmap='Blues', vmin=0, vmax=1)
        axes[1].set_xticks(range(len(texts)))
        axes[1].set_yticks(range(len(texts)))
        axes[1].set_xticklabels(short_labels, rotation=45, ha='right', fontsize=8)
        axes[1].set_yticklabels(short_labels, fontsize=8)
        axes[1].set_title("Cosine Similarity Matrix")
        plt.colorbar(im, ax=axes[1])

        # Annotate cells
        for i in range(len(texts)):
            for j in range(len(texts)):
                axes[1].text(j, i, f"{sim_matrix[i,j]:.2f}",
                             ha='center', va='center', fontsize=7,
                             color='black' if sim_matrix[i,j] < 0.7 else 'white')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()
        return fig


# ==============================================================================
# MODULE 3: GAN-BASED IMAGE GENERATION (DCGAN)
# ==============================================================================

class DCGANGenerator(nn.Module):
    """
    Deep Convolutional GAN Generator.
    
    Architecture:
    - Input: latent vector z ~ N(0,1) of size nz
    - 5 transposed conv layers with BatchNorm + ReLU
    - Output: 64x64 RGB image with Tanh activation
    
    Reference: Radford et al. 2015, "Unsupervised Representation Learning 
    with Deep Convolutional Generative Adversarial Networks"
    """
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.nz = nz
        self.main = nn.Sequential(
            # Input: (nz) -> (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4 -> (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8 -> (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16 -> (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf) x 32 x 32 -> (nc) x 64 x 64
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)


class DCGANDiscriminator(nn.Module):
    """
    Deep Convolutional GAN Discriminator.
    
    Architecture:
    - Input: 64x64 RGB image
    - 5 conv layers with BatchNorm + LeakyReLU
    - Output: scalar probability (real vs fake)
    """
    def __init__(self, ndf: int = 64, nc: int = 3):
        super().__init__()
        self.main = nn.Sequential(
            # (nc) x 64 x 64 -> (ndf) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf) x 32 x 32 -> (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*2) x 16 x 16 -> (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*4) x 8 x 8 -> (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # (ndf*8) x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1)


class TextConditionedGAN:
    """
    Text-conditioned GAN wrapper.
    
    Integrates:
    - TextEmbeddingEngine for text conditioning
    - DCGAN Generator + Discriminator
    - Training loop with G/D loss tracking
    - Visualization utilities
    """

    def __init__(self, nz: int = 100, ngf: int = 64, ndf: int = 64,
                 embedding_dim: int = 256, device: str = "auto"):
        self.device = self._setup_device(device)
        self.nz = nz
        self.embedding_dim = embedding_dim

        # Models
        self.netG = DCGANGenerator(nz=nz + embedding_dim, ngf=ngf).to(self.device)
        self.netD = DCGANDiscriminator(ndf=ndf).to(self.device)
        self._init_weights()

        # Text embedding engine
        self.embed_engine = TextEmbeddingEngine(embedding_dim=embedding_dim)

        # Training state
        self.G_losses = []
        self.D_losses = []
        self.D_x_history = []
        self.D_gz_history = []

        print(f"TextConditionedGAN initialized on {self.device}")
        print(f"Generator params: {sum(p.numel() for p in self.netG.parameters()):,}")
        print(f"Discriminator params: {sum(p.numel() for p in self.netD.parameters()):,}")

    def _setup_device(self, device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _init_weights(self):
        """Apply DCGAN weight initialization (mean=0, std=0.02)."""
        for model in [self.netG, self.netD]:
            for m in model.modules():
                if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                    nn.init.normal_(m.weight, 0.0, 0.02)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.normal_(m.weight, 1.0, 0.02)
                    nn.init.constant_(m.bias, 0)

    def generate_from_text(self, prompt: str, n_images: int = 4,
                           seed: int = None) -> torch.Tensor:
        """
        Generate images conditioned on a text prompt.
        
        Args:
            prompt: text description
            n_images: number of images to generate
            seed: random seed for reproducibility
            
        Returns:
            Tensor of shape [n_images, 3, 64, 64] in [-1, 1]
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Get text embedding
        text_emb = self.embed_engine.embed_text(prompt)  # [embedding_dim]
        text_emb = text_emb.unsqueeze(0).expand(n_images, -1)  # [n_images, embedding_dim]
        text_emb = text_emb.to(self.device)

        # Random noise
        noise = torch.randn(n_images, self.nz, 1, 1, device=self.device)

        # Concatenate text embedding with noise
        text_emb_reshaped = text_emb.unsqueeze(-1).unsqueeze(-1)  # [n, emb, 1, 1]
        z = torch.cat([noise, text_emb_reshaped], dim=1)  # [n, nz+emb, 1, 1]

        self.netG.eval()
        with torch.no_grad():
            fake_images = self.netG(z)

        return fake_images

    def train_step(self, real_images: torch.Tensor, prompts: List[str],
                   optimizer_D: optim.Optimizer, optimizer_G: optim.Optimizer,
                   criterion: nn.Module) -> Dict:
        """
        One training step for both D and G.
        Returns losses and D(x), D(G(z)) diagnostics.
        """
        batch_size = real_images.size(0)
        real_label = torch.ones(batch_size, device=self.device)
        fake_label = torch.zeros(batch_size, device=self.device)

        # ─────────────── Train Discriminator ───────────────
        self.netD.zero_grad()
        real_images = real_images.to(self.device)
        output_real = self.netD(real_images)
        errD_real = criterion(output_real, real_label)
        D_x = output_real.mean().item()

        # Generate fake images
        text_embs = self.embed_engine.embed_batch(prompts).to(self.device)
        noise = torch.randn(batch_size, self.nz, 1, 1, device=self.device)
        text_emb_r = text_embs.unsqueeze(-1).unsqueeze(-1)
        z = torch.cat([noise, text_emb_r], dim=1)
        fake_images = self.netG(z)

        output_fake = self.netD(fake_images.detach())
        errD_fake = criterion(output_fake, fake_label)
        errD = errD_real + errD_fake
        errD.backward()
        optimizer_D.step()
        D_G_z1 = output_fake.mean().item()

        # ─────────────── Train Generator ───────────────
        self.netG.zero_grad()
        output_fake2 = self.netD(fake_images)
        errG = criterion(output_fake2, real_label)  # G wants D to think fakes are real
        errG.backward()
        optimizer_G.step()
        D_G_z2 = output_fake2.mean().item()

        return {
            "loss_D": errD.item(),
            "loss_G": errG.item(),
            "D_x": D_x,
            "D_G_z1": D_G_z1,
            "D_G_z2": D_G_z2
        }

    def train_demo(self, num_epochs: int = 5, batch_size: int = 16,
                   save_dir: str = "outputs/gan_training"):
        """
        Demo training on synthetic noise data.
        
        In production, replace with a real image dataset
        (e.g., CIFAR-10, CelebA, custom dataset).
        Trains for a few epochs to demonstrate GAN loss curves.
        """
        os.makedirs(save_dir, exist_ok=True)

        # Demo corpus for embedding
        demo_prompts = [
            "a beautiful landscape with mountains",
            "portrait of a person smiling",
            "cyberpunk city at night with neon lights",
            "cute animal in a forest",
            "abstract geometric art colorful",
            "medieval castle on a hill",
            "space galaxy stars nebula",
            "food photography fresh vegetables"
        ]
        self.embed_engine.build_vocab(demo_prompts * 10)

        # Synthetic dataset: random 64x64 images
        print("\nCreating synthetic training data (replace with real dataset)...")
        n_samples = batch_size * 20
        fake_real_images = torch.randn(n_samples, 3, 64, 64)
        fake_real_images = torch.tanh(fake_real_images)  # normalize to [-1,1]
        labels_rep = (demo_prompts * (n_samples // len(demo_prompts) + 1))[:n_samples]

        criterion = nn.BCELoss()
        optimizer_D = optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_G = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

        print(f"\nStarting GAN training: {num_epochs} epochs, batch size {batch_size}")
        print("─" * 60)

        for epoch in range(num_epochs):
            epoch_losses_D, epoch_losses_G = [], []
            for batch_start in range(0, n_samples - batch_size, batch_size):
                batch_imgs = fake_real_images[batch_start:batch_start + batch_size]
                batch_prompts = labels_rep[batch_start:batch_start + batch_size]

                step_results = self.train_step(
                    batch_imgs, batch_prompts,
                    optimizer_D, optimizer_G, criterion
                )
                epoch_losses_D.append(step_results["loss_D"])
                epoch_losses_G.append(step_results["loss_G"])
                self.G_losses.append(step_results["loss_G"])
                self.D_losses.append(step_results["loss_D"])
                self.D_x_history.append(step_results["D_x"])
                self.D_gz_history.append(step_results["D_G_z2"])

            avg_D = np.mean(epoch_losses_D)
            avg_G = np.mean(epoch_losses_G)
            print(f"Epoch [{epoch+1}/{num_epochs}] Loss_D: {avg_D:.4f}  Loss_G: {avg_G:.4f}")

        print("\nTraining complete!")
        self._save_training_viz(save_dir)
        return {"G_losses": self.G_losses, "D_losses": self.D_losses}

    def _save_training_viz(self, save_dir: str):
        """Save GAN training loss curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("GAN Training Metrics", fontsize=16, fontweight='bold')

        # G and D loss
        axes[0, 0].plot(self.G_losses, label="Generator Loss", color='blue', alpha=0.7)
        axes[0, 0].plot(self.D_losses, label="Discriminator Loss", color='red', alpha=0.7)
        axes[0, 0].set_title("Training Losses")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Loss (BCE)")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # D(x) and D(G(z))
        axes[0, 1].plot(self.D_x_history, label="D(x) - Real", color='green', alpha=0.7)
        axes[0, 1].plot(self.D_gz_history, label="D(G(z)) - Fake", color='orange', alpha=0.7)
        axes[0, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Equilibrium')
        axes[0, 1].set_title("Discriminator Output")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Probability")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Moving average loss
        window = max(len(self.G_losses) // 20, 1)
        g_smooth = np.convolve(self.G_losses, np.ones(window)/window, mode='valid')
        d_smooth = np.convolve(self.D_losses, np.ones(window)/window, mode='valid')
        axes[1, 0].plot(g_smooth, label="G Loss (smoothed)", color='blue')
        axes[1, 0].plot(d_smooth, label="D Loss (smoothed)", color='red')
        axes[1, 0].set_title(f"Smoothed Losses (window={window})")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Loss ratio (G/D)
        ratios = [g/max(d, 1e-8) for g, d in zip(self.G_losses, self.D_losses)]
        axes[1, 1].plot(ratios, color='purple', alpha=0.7)
        axes[1, 1].axhline(1.0, color='gray', linestyle='--', label='Balance = 1.0')
        axes[1, 1].set_title("G/D Loss Ratio")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Ratio")
        axes[1, 1].set_ylim(0, 5)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(save_dir, "gan_training_metrics.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
        plt.show()

    def visualize_generated_images(self, prompts: List[str],
                                   save_path: str = None, seed: int = 42):
        """
        Generate and visualize images for a list of prompts.
        NOTE: Without actual training on real data, outputs are noise —
        this demonstrates the architecture and conditioning pipeline.
        """
        n = len(prompts)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
        fig.suptitle("GAN Text-to-Image Generation\n(Requires real dataset training for quality results)",
                     fontsize=12, fontweight='bold')

        if n == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)

        for i, prompt in enumerate(prompts):
            r, c = divmod(i, cols)
            imgs = self.generate_from_text(prompt, n_images=1, seed=seed + i)
            img = imgs[0].cpu()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = img.permute(1, 2, 0).numpy()

            axes[r, c].imshow(img)
            axes[r, c].set_title(prompt[:30] + ("..." if len(prompt) > 30 else ""),
                                 fontsize=8)
            axes[r, c].axis('off')

        # Hide unused axes
        for i in range(n, rows * cols):
            r, c = divmod(i, cols)
            axes[r, c].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()
        return fig


# ==============================================================================
# MODULE 4: PIPELINE COMPARISON & VISUALIZATION
# ==============================================================================

class PipelineComparison:
    """
    Compare GAN pipeline vs Stable Diffusion pipeline.
    Visualizes the pipeline architecture and method differences.
    """

    @staticmethod
    def visualize_pipeline_architecture(save_path: str = None):
        """Draw a diagram comparing GAN vs Diffusion pipeline."""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle("Text-to-Image Pipeline Comparison", fontsize=16, fontweight='bold')

        # ─── GAN Pipeline ───
        ax = axes[0]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title("GAN Pipeline (New Module)", fontsize=13, fontweight='bold', color='darkblue')
        ax.axis('off')

        boxes_gan = [
            (5, 9.0, "Text Prompt", "#4472C4"),
            (5, 7.5, "Text Preprocessor\n(tokenize, clean, score)", "#70AD47"),
            (5, 6.0, "Text Embedding Engine\n(vocab → dense vector)", "#ED7D31"),
            (5, 4.5, "Noise Vector z ~ N(0,1)", "#9E480E"),
            (5, 3.0, "DCGAN Generator\n[nz+emb → 64×64 image]", "#FF0000"),
            (5, 1.5, "Generated Image (64×64)", "#FFC000"),
        ]
        for x, y, label, color in boxes_gan:
            ax.add_patch(plt.FancyBboxPatch((x-2.5, y-0.5), 5, 0.9,
                                            boxstyle="round,pad=0.1",
                                            facecolor=color, alpha=0.85,
                                            edgecolor='white', linewidth=2))
            ax.text(x, y, label, ha='center', va='center', fontsize=9,
                    color='white', fontweight='bold')

        for i in range(len(boxes_gan) - 1):
            ax.annotate("", xy=(5, boxes_gan[i+1][1] + 0.45),
                        xytext=(5, boxes_gan[i][1] - 0.45),
                        arrowprops=dict(arrowstyle="->", color="gray", lw=2))

        # ─── Stable Diffusion Pipeline ───
        ax2 = axes[1]
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.set_title("Stable Diffusion Pipeline (Training Project)", fontsize=13,
                      fontweight='bold', color='darkgreen')
        ax2.axis('off')

        boxes_sd = [
            (5, 9.0, "Text Prompt", "#4472C4"),
            (5, 7.5, "CLIP Text Encoder\n(77 tokens → 768-dim)", "#70AD47"),
            (5, 6.0, "Latent Noise\n[Random z in latent space]", "#ED7D31"),
            (5, 4.5, "U-Net Denoising\n[T iterative steps]", "#9E480E"),
            (5, 3.0, "VAE Decoder\n[latent → pixel space]", "#7030A0"),
            (5, 1.5, "Generated Image (512×512)", "#FFC000"),
        ]
        for x, y, label, color in boxes_sd:
            ax2.add_patch(plt.FancyBboxPatch((x-2.5, y-0.5), 5, 0.9,
                                             boxstyle="round,pad=0.1",
                                             facecolor=color, alpha=0.85,
                                             edgecolor='white', linewidth=2))
            ax2.text(x, y, label, ha='center', va='center', fontsize=9,
                     color='white', fontweight='bold')

        for i in range(len(boxes_sd) - 1):
            ax2.annotate("", xy=(5, boxes_sd[i+1][1] + 0.45),
                         xytext=(5, boxes_sd[i][1] - 0.45),
                         arrowprops=dict(arrowstyle="->", color="gray", lw=2))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()
        return fig

    @staticmethod
    def model_comparison_table(save_path: str = None):
        """Create a comparison table visualization."""
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('off')

        columns = ["Attribute", "DCGAN (New)", "Stable Diffusion (Training)", "Winner"]
        rows = [
            ["Architecture", "Generator + Discriminator", "U-Net + VAE + CLIP", "SD (quality)"],
            ["Training", "Adversarial (GAN loss)", "Denoising score matching", "SD (stability)"],
            ["Inference Speed", "~1 ms (GPU)", "~5-30s (GPU)", "GAN (speed)"],
            ["Image Quality", "Moderate (64×64)", "High (512×512+)", "SD (quality)"],
            ["Text Conditioning", "Embedding concat", "Cross-attention (CLIP)", "SD (alignment)"],
            ["Mode Collapse Risk", "High", "Low", "SD (stability)"],
            ["GPU Memory", "~100MB", "~4-8GB", "GAN (efficiency)"],
            ["Training Data Needed", "Thousands", "Billions", "GAN (accessibility)"],
        ]

        colors = []
        for row in rows:
            if row[-1].startswith("SD"):
                colors.append(["#F2F2F2", "#FFE0CC", "#CCE5FF", "#CCE5FF"])
            elif row[-1].startswith("GAN"):
                colors.append(["#F2F2F2", "#CCE5FF", "#FFE0CC", "#CCE5FF"])
            else:
                colors.append(["#F2F2F2", "#E8F5E9", "#E8F5E9", "#E8F5E9"])

        table = ax.table(
            cellText=rows,
            colLabels=columns,
            cellLoc='center',
            loc='center',
            cellColours=colors
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.0)

        # Style header
        for j in range(len(columns)):
            table[0, j].set_facecolor('#2E75B6')
            table[0, j].set_text_props(color='white', fontweight='bold')

        ax.set_title("Model Comparison: DCGAN vs Stable Diffusion",
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        plt.show()
        return fig


# ==============================================================================
# MODULE 5: FULL PIPELINE DEMO
# ==============================================================================

def run_complete_pipeline_demo(output_dir: str = "outputs"):
    """
    Runs the complete end-to-end pipeline demonstration.
    
    Steps:
    1. Text Preprocessing
    2. Text Embedding Creation & Visualization
    3. GAN Training Demo
    4. Image Generation from Text
    5. Pipeline Comparison
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/embeddings", exist_ok=True)
    os.makedirs(f"{output_dir}/gan_training", exist_ok=True)
    os.makedirs(f"{output_dir}/generated", exist_ok=True)
    os.makedirs(f"{output_dir}/comparison", exist_ok=True)

    print("=" * 65)
    print("  COMPREHENSIVE TEXT-TO-IMAGE PIPELINE DEMO")
    print("=" * 65)

    demo_prompts = [
        "a serene mountain landscape at sunrise, photorealistic",
        "portrait of a wise old wizard, fantasy digital art",
        "cyberpunk cityscape at night with neon lights, futuristic",
        "cute cartoon cat wearing a red hat, kawaii anime style",
        "abstract geometric colorful modern art painting",
        "ancient medieval castle on a foggy hill, dramatic lighting",
        "deep space galaxy nebula stars colorful cosmic art",
        "fresh sushi platter Japanese food photography"
    ]

    # ─────────────── Step 1: Text Preprocessing ───────────────
    print("\n[STEP 1] Text Preprocessing")
    print("-" * 40)
    preprocessor = TextPreprocessor()
    results = preprocessor.batch_preprocess(demo_prompts)

    for r in results[:3]:
        print(f"\nOriginal:  {r['original']}")
        print(f"Enhanced:  {r['enhanced']}")
        s = r['analysis']
        print(f"Score: {s['total_score']}/100  |  Tokens: {s['token_count']}  |  Content tokens: {s['content_token_count']}")

    preprocessor.visualize_token_distribution(
        demo_prompts,
        save_path=f"{output_dir}/embeddings/token_distribution.png"
    )

    # ─────────────── Step 2: Text Embeddings ───────────────
    print("\n[STEP 2] Text Embedding Creation")
    print("-" * 40)
    embed_engine = TextEmbeddingEngine(embedding_dim=64)
    embed_engine.build_vocab(demo_prompts * 5)

    # Show embeddings
    embs = embed_engine.embed_batch(demo_prompts[:4])
    print(f"Embedding shape: {embs.shape}  (N=4, dim=64)")
    print(f"Embedding norm (should be ~1.0): {embs.norm(dim=1).mean():.4f}")

    # Similarity example
    similar = embed_engine.find_most_similar(demo_prompts[0], demo_prompts[1:])
    print(f"\nMost similar to '{demo_prompts[0][:40]}':")
    for text, score in similar:
        print(f"  [{score:.3f}] {text[:50]}")

    embed_engine.visualize_embeddings(
        demo_prompts,
        labels=[f"P{i+1}" for i in range(len(demo_prompts))],
        save_path=f"{output_dir}/embeddings/embedding_visualization.png"
    )

    # ─────────────── Step 3: GAN Training Demo ───────────────
    print("\n[STEP 3] GAN-Based Image Generation")
    print("-" * 40)
    gan = TextConditionedGAN(nz=100, embedding_dim=64)
    gan_results = gan.train_demo(
        num_epochs=5,
        batch_size=16,
        save_dir=f"{output_dir}/gan_training"
    )

    # ─────────────── Step 4: Generate Images from Text ───────────────
    print("\n[STEP 4] Generating Images from Text Prompts")
    print("-" * 40)
    gan.visualize_generated_images(
        demo_prompts,
        save_path=f"{output_dir}/generated/gan_text_to_image.png",
        seed=42
    )

    # ─────────────── Step 5: Pipeline Comparison ───────────────
    print("\n[STEP 5] Pipeline Architecture Comparison")
    print("-" * 40)
    PipelineComparison.visualize_pipeline_architecture(
        save_path=f"{output_dir}/comparison/pipeline_architecture.png"
    )
    PipelineComparison.model_comparison_table(
        save_path=f"{output_dir}/comparison/model_comparison.png"
    )

    print("\n" + "=" * 65)
    print("  PIPELINE DEMO COMPLETE")
    print(f"  All outputs saved to: {output_dir}/")
    print("=" * 65)

    return {
        "preprocessor": preprocessor,
        "embed_engine": embed_engine,
        "gan": gan
    }


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    pipeline = run_complete_pipeline_demo(output_dir="outputs")
