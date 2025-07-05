import argparse
from torchvision import transforms
from datasets import load_dataset
from dataset import HuggingFace
from torch.utils.data import DataLoader
import wandb
import optax
from vqvae import VQVAE


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=61)

    # Dataset
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)

    # VQVAE
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--num-embed', type=int, default=1024)
    parser.add_argument('--commitment-cost', type=float, default=0.25)
    parser.add_argument('--img-channel', type=int, default=3)
    parser.add_argument('--lr-rate', type=float, default=1e-4)
    parser.add_argument('--num-epochs', type=int, default=100)

    # Wandb
    parser.add_argument('--project', type=str, default='VQ-VAE')
    parser.add_argument('--name', type=str, default='run_standard')

    # Save
    parser.add_argument('--checkpoint-path', type=str, required=True)

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

    model = VQVAE(
        embedding_dim=args.embed_dim,
        num_embeddings=args.num_embed,
        commitment_cost=args.commitment_cost,
        output_channels=args.img_channel,
        channel_multipliers=[1, 1, 2, 2, 4],
    )

    optimizer = optax.chain(optax.adam(
        learning_rate=args.lr_rate,
        b1=0.9,
        b2=0.96,
    ))

    epochs = args.num_epochs

    run = wandb.init(
        project=args.project,
        name=args.name,
        reinit=True,
        config=vars(args)
    )

    return {
        'seed': args.seed,
        'train_loader': train_loader,
        'model': model,
        'optimizer': optimizer,
        'epochs': epochs,
        'run': run,
        'checkpoint_path': args.checkpoint_path,
    }