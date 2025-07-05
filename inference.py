import argparse
import sys

import jax
import jax.numpy as jnp
from vqvae import VQVAE
from maskgit import MaskGit
import utils
import torch
import numpy as np

from torchvision.utils import save_image


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=61)

    parser.add_argument('--num_samples', type=int, default=32)

    # VQVAE
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--num-embed', type=int, default=1024)

    # MaskGit
    parser.add_argument('--num-codebook', type=int, default=1024)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--t-embed-dim', type=int, default=768)

    # Token ids
    parser.add_argument('--mask-token-id', type=int, default=1025)
    parser.add_argument('--sos-token-id', type=int, default=1024)

    # Save
    parser.add_argument('--vqvae-path', type=str, required=True)
    parser.add_argument('--maskgit-path', type=str, required=True)

    return parser.parse_args(args)


class Generator(object):

    def __init__(
            self,
            maskgit,
            maskgit_params,
            vqvae,
            vqvae_params,
            sos_token_id,
            mask_token_id,
            num_image_tokens=256,
            temperature=4.5,
    ):
        self.maskgit = maskgit
        self.maskgit_p = maskgit_params
        self.vqvae = vqvae
        self.vqvae_p = vqvae_params
        self.sos_token_id = sos_token_id
        self.mask_token_id = mask_token_id
        self.num_image_tokens = num_image_tokens
        self.temperature = temperature
        self.resolution = self.num_image_tokens // 16

    def create_inputs_tokens(self, num_samples):
        sos_tokens = self.sos_token_id * jnp.ones((num_samples, 1), dtype=jnp.int32)
        mask_tokens = self.mask_token_id * jnp.ones((num_samples, self.num_image_tokens))

        inputs_tokens = jnp.concatenate([sos_tokens, mask_tokens], axis=1)  # [N, seq_len]

        return inputs_tokens.astype(int)

    def generate_samples(self, rng, num_iter=12, num_samples=8):
        def tokens_to_logits(ids):
            return self.maskgit.apply(self.maskgit_p, ids, train=False)

        inputs = self.create_inputs_tokens(num_samples)  # [N, seq_len]
        z_indices = utils.decode(inputs, rng, tokens_to_logits, self.mask_token_id, num_iter, self.temperature)  # [N, num_iter, seq_len]

        z_indices = jnp.reshape(z_indices[:, -1, 1:], (-1, self.resolution, self.resolution))  # [N, 16, 16]

        gen_images = self.vqvae.apply(self.vqvae_p, z_indices, method=self.vqvae.decode)  # [N, H, W, 3]

        return gen_images


def main(args):
    args = parse_args(args)

    vqvae = VQVAE([1, 1, 2, 2, 4])
    maskgit = MaskGit(
        num_codebook=args.num_codebook,
        n_heads=args.num_heads,
        n_layers=args.num_layers,
        embed_dim=args.t_embed_dim,
    )

    key = jax.random.PRNGKey(args.seed)

    vqvae_params = utils.load_params(args.vqvae_path)
    maskgit_params = utils.load_params(args.maskgit_path)

    gen = Generator(maskgit, maskgit_params, vqvae, vqvae_params, args.sos_token_id, args.mask_token_id)

    x_gen = gen.generate_samples(key, num_samples=args.num_samples)

    for i in range(x_gen.shape[0]):
        img = np.array(x_gen[i])

        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img)

        save_image(img, f'gen_images/generated_image{i}.png')


if __name__ == '__main__':
    main(sys.argv[1:])
