# MaskGIT: Masked Generative Image Transformer from Scratch in JAX/FLAX

This repository contains a from-scratch implementation of the paper:

> ** MaskGIT: Masked Generative Image Transformer**  
> (https://arxiv.org/abs/2202.04200)

## 🏁 Training VQVAE

```bash
python vqvae_train.py config.py --checkpoint-path path/to/vqvae.pkl
```

## 🏁 Training MaskGit

```bash
python maskgit_train.py config.py --checkpoint-path path/to/vqvae.pkl --maskgit-path path/to/maskgit.pkl
```

## 🎨 Inference

```bash
python inference.py --vqvae-path path/to/vqvae.pkl --maskgit-path path/to/maskgit.pkl
```

## 🖼 Sample Generated Images From CelebA

![Generated Image](gen_images/generated_image17.png)
![Generated Image](gen_images/generated_image20.png)
![Generated Image](gen_images/generated_image21.png)
![Generated Image](gen_images/generated_image26.png)
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
![Generated Image](gen_images/generated_image85.png)
![Generated Image](gen_images/generated_image87.png)
![Generated Image](gen_images/generated_image90.png)
![Generated Image](gen_images/generated_image93.png)
![Generated Image](gen_images/generated_image101.png)
![Generated Image](gen_images/generated_image120.png)
![Generated Image](gen_images/generated_image127.png)
