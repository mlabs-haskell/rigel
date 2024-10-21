from .context_vector_loader import ContextVectorDataLoader
from .hierarchical_compression import train_compression_network

import fire
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import tqdm

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def train(
    checkpoint_file: str,
    network_type: str,
    batch_size: int = 150,
    cvdb_folder: str = "context_vectors",
    tfidf_file: str = "tfidf.json",
    epochs: int = 100,
    reduction_factor: int | None = None,
    device: str = DEVICE
):
    """Function to train a hierarchical compression network. Saves model after
    each epoch in checkpoint_file. If checkpoint_file already exists, training
    will resume from last saved epoch"""
    with torch.device(device):
        torch.set_default_dtype(torch.float32)

        train_loader = ContextVectorDataLoader(batch_size, tfidf_file, 'train', cvdb_folder)
        val_loader = ContextVectorDataLoader(batch_size, tfidf_file, 'val', cvdb_folder)
        train_compression_network(
            train_loader,
            val_loader,
            epochs,
            checkpoint_file,
            network_type,
            reduction_factor=reduction_factor
        )

def min_loss(checkpoint_file: str = "model.pt"):
    """Function to identify the epoch where the minimum validation loss occurred
    """
    checkpoint_path = Path(checkpoint_file)
    if checkpoint_path.is_file():
        # Load epoch losses
        checkpoint = torch.load(checkpoint_path)
        epoch_losses = checkpoint['losses']

        if len(epoch_losses) > 0:
            min_idx, min_loss = min(enumerate(epoch_losses), key=lambda t: t[1])
            print(f"Min loss of {min_loss} found after {min_idx + 1} epochs")

            plt.plot(epoch_losses)
            plt.show()

        else:
            print("No epoch history found")

    else:
        print(f"Error: could not find file {checkpoint_file}")
        exit(1)

def count_ys():
    """Function to count up how many targets are 0 vs how many are not
    """
    loader = ContextVectorDataLoader(150, "tfidf.json", 'train', "context_vectors")
    zeros = 0
    others = 0

    cos_sims = []

    pbar = tqdm.tqdm(loader)
    for _, y in pbar:
        pbar.set_description(f"{zeros} zeros and {others} others")
        m = y != 0.0
        num_other = torch.count_nonzero(m)
        num_zeros = y.nelement() - num_other
        others += num_other
        zeros += num_zeros
        cos_sims += y.flatten().tolist()

    plt.hist(cos_sims, 20, (0.0, 1.0))
    plt.title("Distribution of Cosine Similarities")
    plt.show()

    print(f"Num zeros: {zeros}")
    print(f"Num other: {others}")

if __name__ == "__main__":
    fire.Fire({
        'train': train,
        'min_loss': min_loss,
        'count_ys': count_ys
    })
