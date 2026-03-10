# MaskGIT — JAX/Flax Implementation

A from-scratch implementation of **MaskGIT: Masked Generative Image Transformer** ([Chang et al., 2022](https://arxiv.org/abs/2202.04200)) in JAX/Flax, including a fully custom VQGAN tokenizer trained from scratch.

---

## What is MaskGIT?

MaskGIT is a two-stage generative model. First, a VQGAN encodes images as discrete token sequences. Then, a bidirectional transformer is trained with a masked token prediction objective — similar to BERT, but for image tokens. At inference, generation starts from a fully masked sequence and iteratively fills in tokens over multiple steps, guided by a confidence-based masking schedule.

---

## Implemented Components

### Stage 1 — VQGAN Tokenizer
| Component | Description |
|---|---|
| **VQ-VAE** | Encoder/decoder with vector quantized bottleneck |
| **Patch discriminator** | Adversarial loss for sharp reconstructions |
| **LPIPS perceptual loss** | Feature-level perceptual similarity loss |
| **Full two-stage tokenizer** | Trained entirely from scratch, no pretrained weights |

### Stage 2 — Masked Transformer
| Component | Description |
|---|---|
| **Bidirectional transformer** | BERT-style encoder operating on image tokens |
| **Masked token prediction** | Training with cosine-scheduled variable masking ratio |
| **Confidence-based sampling** | Tokens with highest predicted confidence unmasked first |
| **Iterative decoding** | Multi-step generation from fully masked sequence |

---

## Training Details

| Setting | Value |
|---|---|
| Dataset | ImageNet / CelebA |
| Framework | JAX/Flax |
| Accelerator | Google Colab |
| Stage 1 | VQGAN trained from scratch |
| Stage 2 | Masked transformer on frozen VQGAN tokens |

---

## Generated Samples

<!-- Replace this with your actual image grid -->
![Generated Image](gen_images/generated_image17.png)
![Generated Image](gen_images/generated_image20.png)
![Generated Image](gen_images/generated_image21.png)
![Generated Image](gen_images/generated_image27.png)
![Generated Image](gen_images/generated_image39.png)
![Generated Image](gen_images/generated_image42.png)
![Generated Image](gen_images/generated_image47.png)
![Generated Image](gen_images/generated_image49.png)
![Generated Image](gen_images/generated_image55.png)
![Generated Image](gen_images/generated_image75.png)
![Generated Image](gen_images/generated_image76.png)
![Generated Image](gen_images/generated_image80.png)
![Generated Image](gen_images/generated_image83.png)
![Generated Image](gen_images/generated_image87.png)
![Generated Image](gen_images/generated_image90.png)
![Generated Image](gen_images/generated_image101.png)
![Generated Image](gen_images/generated_image120.png)
![Generated Image](gen_images/generated_image127.png)

---

## Implementation Notes

Non-trivial details reproduced faithfully from the paper:

- The masking ratio during training is sampled from a cosine schedule distribution, not fixed — this is crucial for the model to learn to decode under any masking level
- At inference, tokens are revealed following the cosine schedule: more tokens per step early, fewer later
- The transformer uses **bidirectional (non-causal) attention** — a critical distinction from autoregressive models like GPT
- LPIPS loss uses frozen VGG features; gradients flow through the decoder but not the perceptual network
- The discriminator is only activated after a warmup period to stabilize early VAE training

---

## Project Structure
```
MaskGit/
├── vqvae.py               # VQ-VAE encoder, decoder, quantizer
├── vqvae_config.py        # VQGAN hyperparameters
├── discriminator.py       # PatchGAN discriminator
├── lpips.py               # Perceptual loss (LPIPS)
├── train_vqvae.py         # Stage 1: VQGAN training
├── maskgit.py             # Bidirectional transformer + masking
├── maskgit_config.py      # MaskGIT hyperparameters
├── train_maskgit.py       # Stage 2: Masked transformer training
├── dataset.py             # Data loading and preprocessing
├── utils.py               # Shared utilities
└── inference.py           # Generation script
```

---

## References
```bibtex
@inproceedings{chang2022maskgit,
  title={MaskGIT: Masked Generative Image Transformer},
  author={Chang, Huiwen and others},
  booktitle={CVPR},
  year={2022}
}
```

**Official Implementation:** [google-research/maskgit](https://github.com/google-research/maskgit)




