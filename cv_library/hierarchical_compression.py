import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from loss_functions import CosineSimilarityLoss, SequenceLoss

import math
from pathlib import Path
import tqdm
from typing import Literal

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

class HierarchicalAttention(nn.Module):
    def __init__(self, cv_size: torch.Size, reduction: int = 8):
        super().__init__()

        stack = []
        self.freqs_cis = []
        curr_features = cv_size[-1]
        seq_len = cv_size[-2]
        while curr_features > Attention.NUM_HEADS and curr_features > reduction:
            freqs_cis = precompute_freqs_cis(curr_features // Attention.NUM_HEADS, seq_len)
            self.freqs_cis.append(freqs_cis)

            out_features = curr_features // reduction
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

class HierarchicalLinear(nn.Module):
    def __init__(self, cv_size: torch.Size, first_layer_out: int = 512, reduction: int = 16):
        super().__init__()

        curr_features = cv_size[-1] * cv_size[-2]
        stack = [nn.Linear(curr_features, first_layer_out)]
        curr_features = first_layer_out
        print(f"Layer #{len(stack)} output size: {curr_features}")
        while curr_features > reduction:
            out_features = curr_features // reduction
            stack.append(nn.Linear(curr_features, out_features))
            curr_features = out_features

            print(f"Layer #{len(stack)} output size: {curr_features}")

        self.stack = nn.Sequential(*stack)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = x.flatten(-2)

        outputs = []
        for layer in self.stack:
            x = layer.forward(x)
            x = F.tanh(x)
            outputs.append(x)

        return outputs

def run_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    network: HierarchicalAttention | HierarchicalLinear,
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
            loss_X1 = compressed_vector[cv1_idx]
            for cv2_idx in range(cv1_idx + 1, len(compressed_vector), loss_batch_size):
                # Get the Xs and y for the loss calculation
                loss_X2 = compressed_vector[cv2_idx : cv2_idx + loss_batch_size]
                loss_y = y[cv1_idx, cv2_idx : cv2_idx + loss_batch_size]

                # Calculate the loss
                batch_loss += factor * loss_fn(loss_X1, loss_X2, loss_y)

    return batch_loss

def train_compression_network(
    train_dataset: DataLoader,
    val_dataset: DataLoader,
    epochs: int,
    checkpoint_file: str,
    network_type: Literal["attention", "linear"],
    loss_batch_size: int = 100,
    reduction_factor: int | None = None
) -> HierarchicalAttention:
    # Set up the network
    kwargs = {'cv_size': [128, 4096]}
    if reduction_factor is not None:
        kwargs['reduction'] = reduction_factor
    match network_type:
        case "attention":
            network = HierarchicalAttention(**kwargs)
            loss_fn = SequenceLoss(y_scale=2)
        case "linear":
            network = HierarchicalLinear(**kwargs)
            loss_fn = CosineSimilarityLoss(y_scale=2)
        case _:
            raise ValueError(
                f"train_compression_network: {network_type} not a recognized network type"
            )
    optimizer = Adam(network.parameters())

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
    for _ in epoch_pbar:
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