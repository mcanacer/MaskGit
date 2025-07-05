MaskGIT: Masked Generative Image Transformer from Scratch in JAX/FLAX
https://arxiv.org/abs/2202.04200

Training VQVAE

python vqvae_train.py config.py --checkpoint-path path/to/vqvae.npz

Training MaskGit

python maskgit_train.py config.py --checkpoint-path path/to/vqvae.npz --maskgit-path path/to/maskgit.npz --seq-path path/to/seq.npz

Inference

python inference.py --vqvae-path path/to/vqvae.npz --maskgit-path path/to/maskgit.npz
