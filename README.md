# MaskGIT: Masked Generative Image Transformer from Scratch in JAX/FLAX

This repository contains a from-scratch implementation of the paper:

> ** MaskGIT: Masked Generative Image Transformer**  
> (https://arxiv.org/abs/2202.04200)

## 🏁 Training VQVAE

```bash
python vqvae_train.py config.py --checkpoint-path path/to/vqvae.npz
```

## 🏁 Training MaskGit

```bash
python maskgit_train.py config.py --checkpoint-path path/to/vqvae.npz --maskgit-path path/to/maskgit.npz --seq-path path/to/seq.npz
```

## 🎨 Inference

```bash
python inference.py --vqvae-path path/to/vqvae.npz --maskgit-path path/to/maskgit.npz
```

## 🖼 Sample Generated Images From CelebA

![Generated Image](gen_images/generated_image3.png)
![Generated Image](gen_images/generated_image4.png)
![Generated Image](gen_images/generated_image16.png)
![Generated Image](gen_images/generated_image20.png)
![Generated Image](gen_images/generated_image22.png)
![Generated Image](gen_images/generated_image24.png)
