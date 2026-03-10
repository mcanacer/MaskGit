# MaskGIT — JAX/Flax Implementation

A from-scratch implementation of **MaskGIT: Masked Generative Image Transformer** ([Chang et al., 2022](https://arxiv.org/abs/2202.04200)) in JAX/Flax.

---

## What is MaskGIT?

MaskGIT is a two-stage generative model. First, a VQGAN encodes images as discrete token sequences. Then, a bidirectional transformer is trained with a masked token prediction objective — similar to BERT, but for image tokens. At inference, generation starts from a fully masked sequence and iteratively fills in tokens over multiple steps, guided by a confidence-based masking schedule.

---

## Implemented Components

| Component | Description |
|---|---|
| **VQGAN tokenizer** | Encodes images into discrete codebook indices |
| **Bidirectional transformer** | BERT-style encoder operating on image tokens |
| **Masked token prediction** | Training with variable masking ratio |
| **Cosine masking schedule** | Inference masking schedule for iterative decoding |
| **Confidence-based sampling** | Tokens with highest predicted confidence unmasked first |
| **Class conditioning** | Class label conditioning via token prepending |

---

## Training Details

| Setting | Value |
|---|---|
| Dataset | ImageNet / CelebA |
| Framework | JAX/Flax |
| Accelerator | Google Colab |
| Stage 1 | Pretrained VQGAN tokenizer |
| Stage 2 | Masked transformer on frozen tokens |

---

## Generated Samples

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
- Class conditioning prepends a learned class embedding token to the sequence before the masked image tokens

---

## Project Structure
```
maskgit-jax/
├── tokenizer/
│   └── vqgan.py             # Pretrained VQGAN for tokenization
├── model/
│   ├── transformer.py       # Bidirectional transformer
│   └── embeddings.py        # Token + position + class embeddings
├── masking/
│   ├── schedule.py          # Cosine masking schedule
│   └── strategy.py          # Confidence-based iterative decoding
└── train.py
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






