import argparse
from torchvision import transforms
from datasets import load_dataset
from dataset import HuggingFace
from torch.utils.data import DataLoader
import wandb
import optax
from maskgit import MaskGit
from vqvae import VQVAE


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=61)

    # Dataset
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)

    # MaskGit
    parser.add_argument('--num-codebook', type=int, default=1024)
    parser.add_argument('--num-heads', type=int, default=8)
    parser.add_argument('--num-layers', type=int, default=12)
    parser.add_argument('--embed-dim', type=int, default=768)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr-rate', type=float, default=1e-4)
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    # Wandb
    parser.add_argument('--project', type=str, default='MaskGit')
    parser.add_argument('--name', type=str, default='run_standard')

    # Save
    parser.add_argument('--checkpoint-path', type=str, required=True)
    parser.add_argument('--maskgit-path', type=str, required=True)
    parser.add_argument('--seq-path', type=str, default='/content/drive/MyDrive/VQ-VAE/celeb_sequences.npy')

    return parser.parse_args(args)


def everything(args):
    args = parse_args(args)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),  # Normalize [0, 1]
        transforms.Lambda(lambda t: (t * 2.0) - 1.0),  # Scale [-1, 1]
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # Convert [C, H, W] to [H, W, C]
        transforms.Lambda(lambda x: x.numpy()),
    ])

    train_dataset = HuggingFace(
      dataset=load_dataset("flwrlabs/celeba", split='train'),
      transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    optimizer = optax.chain(optax.adam(
        learning_rate=args.lr_rate,
        b1=0.9,
        b2=0.96,
    ))

    epochs = args.num_epochs

    model = VQVAE(
        channel_multipliers=[1, 1, 2, 2, 4],
    )

    maskgit = MaskGit(
        num_codebook=args.num_codebook,
        n_heads=args.num_heads,
        n_layers=args.num_layers,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
    )

    run = wandb.init(
        project=args.project,
        name=args.name,
        reinit=True,
        config=vars(args)
    )

    return {
        'seed': args.seed,
        'train_loader': train_loader,
        'optimizer': optimizer,
        'epochs': epochs,
        'model': model,
        'maskgit': maskgit,
        'num_codebook': args.num_codebook,
        'run': run,
        'label_smoothing': args.label_smoothing,
        'checkpoint_path': args.checkpoint_path,
        'maskgit_path': args.maskgit_path,
        'seq_path': args.seq_path,
    }