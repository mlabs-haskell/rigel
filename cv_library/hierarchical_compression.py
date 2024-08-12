import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader

from context_vector_loader import ContextVectorDataLoader
from similarity_function import SequenceLoss

import fire
import math
from pathlib import Path
import tqdm

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# NOTE: The Attention-related functions and classes are mostly copied from Llama
# This is because I expect we'll need to customize them. If that turns out not
# to be the case, then we should just import them from the Llama module
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    NUM_HEADS = 32

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.wq = nn.Linear(in_features, in_features, bias=False)
        self.wk = nn.Linear(in_features, in_features, bias=False)
        self.wv = nn.Linear(in_features, in_features, bias=False)
        self.wo = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, in_features = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        head_dim = in_features // self.NUM_HEADS
        xq = xq.view(bsz, seqlen, self.NUM_HEADS, head_dim)
        xk = xk.view(bsz, seqlen, self.NUM_HEADS, head_dim)
        xv = xv.view(bsz, seqlen, self.NUM_HEADS, head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class HierarchicalCompression(nn.Module):
    def __init__(self, cv_size: int):
        super().__init__()

        stack = []
        self.freqs_cis = []
        curr_features = cv_size
        while curr_features > Attention.NUM_HEADS:
            freqs_cis = precompute_freqs_cis(curr_features // Attention.NUM_HEADS, 128)
            self.freqs_cis.append(freqs_cis)

            out_features = curr_features // 8
            stack.append(Attention(curr_features, out_features))
            curr_features = out_features

            print(f"Layer #{len(stack)} output size: {curr_features}")

        self.stack = nn.Sequential(*stack)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for layer, freqs_cis in zip(self.stack, self.freqs_cis):
            x = layer(x, freqs_cis)
            x = F.tanh(x)
            outputs.append(x)

        return outputs

def run_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    network: HierarchicalCompression,
    loss_batch_size: int,
    loss_fn: nn.Module,
    factor: float
) -> torch.Tensor:
    # Push the data through the network
    compressed_vectors = network.forward(X)

    # Calculate loss at each level of compression
    batch_loss = torch.tensor(0.0)
    for compressed_vector in compressed_vectors:
        # Get the vectors for loss calculation
        for cv1_idx in range(len(compressed_vector)):
            X1 = compressed_vector[cv1_idx]
            for cv2_idx in range(cv1_idx + 1, len(compressed_vector), loss_batch_size):
                # Get the Xs and y for the loss calculation
                loss_X2 = compressed_vector[cv2_idx : cv2_idx + loss_batch_size]
                loss_X1 = X1.unsqueeze(0).expand((len(loss_X2), -1, -1))
                loss_y = y[cv1_idx, cv2_idx : cv2_idx + loss_batch_size]

                # Calculate the loss
                batch_loss += factor * loss_fn(loss_X1, loss_X2, loss_y)

    return batch_loss

def train_compression_network(
    train_dataset: DataLoader,
    val_dataset: DataLoader,
    epochs: int,
    checkpoint_file: str,
    loss_batch_size: int = 100
) -> HierarchicalCompression:
    # Set up the network
    network = HierarchicalCompression(4096)
    optimizer = Adam(network.parameters())
    loss_fn = SequenceLoss(y_scale=2)

    # Check if checkpoint already exists
    epoch_losses = []
    checkpoint_path = Path(checkpoint_file)
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path)
        network.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_losses = checkpoint['losses']

    # Iterate through the epochs
    start_epoch = len(epoch_losses)
    epoch_pbar = tqdm.tqdm(range(start_epoch, epochs), leave=False)
    for i in epoch_pbar:
        # Train the model
        batch_pbar = tqdm.tqdm(train_dataset, leave=False, desc="Training")
        network.train()
        for X, y in batch_pbar:
            # Push the data through the network
            optimizer.zero_grad()
            factor = 2.0 ** len(network.stack)
            batch_loss = run_batch(X, y, network, loss_batch_size, loss_fn, factor)

            # Do a step of gradient descent
            batch_loss.backward()
            optimizer.step()
            factor /= 2.0

            # Display the loss for this batch
            disp_loss = batch_loss.item() / (len(X) ** 2)
            batch_pbar.set_postfix({'batch loss': disp_loss})

        # Validate the model
        with torch.no_grad():
            batch_pbar = tqdm.tqdm(val_dataset, leave=False, desc="Validating")
            network.eval()
            total_loss = 0.0
            total_comparisons = 0
            for X, y in batch_pbar:
                # Evaluate the batch
                factor = 2.0 ** len(network.stack)
                batch_loss = run_batch(X, y, network, loss_batch_size, loss_fn, factor)
                factor /= 2.0

                # Increment totals
                total_loss += batch_loss
                total_comparisons += len(X) ** 2

                # Display the loss for this batch
                disp_loss = batch_loss.item() / (len(X) ** 2)
                batch_pbar.set_postfix({'batch loss': disp_loss})

        # Display the loss for this epoch
        disp_loss = total_loss.item() / total_comparisons
        epoch_losses.append(disp_loss)
        epoch_pbar.set_postfix({'validation loss': disp_loss})

        # Save it
        torch.save({
            'model_state_dict': network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': epoch_losses
        }, checkpoint_file)

    return network

def train(
    batch_size: int = 150,
    tfidf_file: str = "../tfidf.json",
    epochs: int = 100,
    checkpoint_file: str = "model.pt"
):
    """Function to train a hierarchical compression network. Saves model after
    each epoch in checkpoint_file. If checkpoint_file already exists, training
    will resume from last saved epoch"""
    with torch.device(DEVICE):
        torch.set_default_dtype(torch.float32)
        train_loader = ContextVectorDataLoader(batch_size, tfidf_file, 'train')
        val_loader = ContextVectorDataLoader(batch_size, tfidf_file, 'val')
        train_compression_network(train_loader, val_loader, epochs, checkpoint_file)

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
        else:
            print("No epoch history found")

    else:
        print(f"Error: could not find file {checkpoint_file}")
        exit(1)

def count_ys():
    """Function to count up how many targets are 0 vs how many are not
    """
    loader = ContextVectorDataLoader(150, "../tfidf.json", 'train')
    zeros = 0
    others = 0

    pbar = tqdm.tqdm(loader)
    for _, y in pbar:
        pbar.set_description(f"{zeros} zeros and {others} others")
        m = y != 0.0
        num_other = torch.count_nonzero(m)
        num_zeros = y.nelement() - num_other
        others += num_other
        zeros += num_zeros

    print(f"Num zeros: {zeros}")
    print(f"Num other: {others}")

if __name__ == "__main__":
    fire.Fire({
        'train': train,
        'min_loss': min_loss,
        'count_ys': count_ys
    })