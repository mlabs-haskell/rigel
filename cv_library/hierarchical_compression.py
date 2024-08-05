import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from context_vector_loader import ContextVectorDataLoader

class HierarchicalCompression(nn.Module):
    def __init__(self, cv_size: int):
        super().__init__()

        stack = []
        curr_features = cv_size
        while curr_features > 1:
            print(curr_features)
            out_features = curr_features // 2048
            stack.append(nn.Sequential(
                nn.Linear(curr_features, out_features),
                nn.Tanh()
            ))
            curr_features = out_features

        self.stack = nn.Sequential(*stack)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for layer in self.stack:
            x = layer(x)
            outputs.append(x)

        return outputs

def train_compression_network(
    context_vectors: DataLoader,
    epochs: int
) -> HierarchicalCompression:
    # Get number of features
    X, _ = next(iter(context_vectors))
    cv_size = X.shape[-1]

    network = HierarchicalCompression(cv_size)
    optimizer = Adam(network.parameters())
    loss_fn = nn.CosineEmbeddingLoss()
    for i in range(epochs):
        epoch_loss = 0.0
        num_elements = 0
        for X, y in context_vectors:
            # Push the data through the network
            optimizer.zero_grad()
            compressed_vectors = network(X)

            # Calculate loss at each level of compression
            # TODO: Maybe add additional weight for each tier?
            batch_loss = torch.tensor(0.0)
            for compressed_vector in compressed_vectors:
                # Get the vectors for loss calculation
                for cv1_idx in range(len(compressed_vector)):
                    for cv2_idx in range(cv1_idx + 1, len(compressed_vector)):
                        loss_X1 = compressed_vector[cv1_idx]
                        loss_X2 = compressed_vector[cv2_idx]
                        loss_y = y[cv1_idx, cv2_idx]
                        batch_loss += loss_fn(loss_X1, loss_X2, loss_y)

            num_elements += len(X)
            epoch_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / num_elements
        print(f"Epoch {i + 1}/{epochs} loss: {avg_loss}")

    return network

with torch.device("cuda"):
    loader = ContextVectorDataLoader(250, "../tfidf.json")
    train_compression_network(loader, 100)