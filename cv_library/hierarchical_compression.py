import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm

from context_vector_loader import ContextVectorDataLoader
from similarity_function import SequenceLoss

import math

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
            print(curr_features)

            freqs_cis = precompute_freqs_cis(curr_features // Attention.NUM_HEADS, 128)
            self.freqs_cis.append(freqs_cis)

            out_features = curr_features // 8
            stack.append(Attention(curr_features, out_features))
            curr_features = out_features

        self.stack = nn.Sequential(*stack)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for layer, freqs_cis in zip(self.stack, self.freqs_cis):
            x = layer(x, freqs_cis)
            x = F.tanh(x)
            outputs.append(x)

        return outputs

LOSS_BATCH_SIZE = 100

def train_compression_network(
    context_vectors: DataLoader,
    epochs: int
) -> HierarchicalCompression:
    network = HierarchicalCompression(4096)
    optimizer = Adam(network.parameters())
    loss_fn = SequenceLoss()
    for i in tqdm.tqdm(range(epochs)):
        epoch_loss = 0.0
        pbar = tqdm.tqdm(context_vectors)
        for X, y in pbar:
            # Push the data through the network
            optimizer.zero_grad()
            compressed_vectors = network.forward(X)

            # Calculate loss at each level of compression
            # TODO: Maybe add additional weight for each tier?
            batch_loss = torch.tensor(0.0)
            for compressed_vector in compressed_vectors:
                # Get the vectors for loss calculation
                for cv1_idx in range(len(compressed_vector)):
                    X1 = compressed_vector[cv1_idx]
                    for cv2_idx in range(cv1_idx + 1, len(compressed_vector), LOSS_BATCH_SIZE):
                        loss_X2 = compressed_vector[cv2_idx : cv2_idx + LOSS_BATCH_SIZE]
                        loss_X1 = X1.unsqueeze(0).expand((len(loss_X2), -1, -1))

                        # Make the loss -1 (to min. cos sim) or 1 (to max.)
                        loss_y = y[cv1_idx, cv2_idx : cv2_idx + LOSS_BATCH_SIZE]
                        #loss_y = 2 * torch.round(y[cv1_idx, cv2_idx]) - 1

                        batch_loss += loss_fn(loss_X1, loss_X2, loss_y)

            epoch_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()
            disp_loss = batch_loss.item() / (len(X) ** 2)
            pbar.set_description(f"Batch loss: {disp_loss:.3f}")

    return network

with torch.device("cuda"):
    torch.set_default_dtype(torch.float32)
    loader = ContextVectorDataLoader(150, "../tfidf.json", device=DEVICE)
    train_compression_network(loader, 100)