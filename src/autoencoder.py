"""Train and use a PyTorch autoencoder to compress CVE embeddings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import faiss
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "autoencoder.pth"
COMPRESSED_PATH = EMBEDDINGS_DIR / "embeddings_compressed.npy"
SOURCE_EMBEDDING_PATHS = [
    EMBEDDINGS_DIR / "embeddings_2023.npy",
    EMBEDDINGS_DIR / "embeddings_2024.npy",
]
EMBEDDING_DIMENSION = 384
HIDDEN_DIMENSION = 128
COMPRESSED_DIMENSION = 64
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MISSING_EMBEDDINGS_MESSAGE = "Run python -m src.embedder --years 2023 2024 first."
MISSING_MODEL_MESSAGE = "Run python -m src.autoencoder --train first."


class CVEAutoencoder(nn.Module):
    """Autoencoder that compresses 384-dimensional embeddings down to 64 dimensions."""

    def __init__(self) -> None:
        """Build the encoder and decoder layers used for compression and reconstruction."""
        super().__init__()

        # The encoder learns a smaller 64-dimensional representation of each embedding.
        self.encoder = nn.Sequential(
            nn.Linear(EMBEDDING_DIMENSION, HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMENSION, COMPRESSED_DIMENSION),
        )

        # The decoder tries to rebuild the original 384-dimensional embedding.
        self.decoder = nn.Sequential(
            nn.Linear(COMPRESSED_DIMENSION, HIDDEN_DIMENSION),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMENSION, EMBEDDING_DIMENSION),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run the full autoencoder: encode first, then decode back to the original size."""
        compressed = self.encoder(inputs)
        return self.decoder(compressed)


def configure_logging() -> None:
    """Configure logging so CLI runs show readable progress messages."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments that choose between training and encoding modes."""
    parser = argparse.ArgumentParser(description="Train or use the CyberSecRAG autoencoder.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", action="store_true", help="Train the autoencoder model.")
    mode_group.add_argument(
        "--encode",
        action="store_true",
        help="Load an existing model and generate compressed embeddings.",
    )
    return parser.parse_args()


def load_combined_embeddings() -> np.ndarray | None:
    """Load and combine the 2023 and 2024 embedding files into one NumPy array."""
    embedding_arrays: list[np.ndarray] = []

    for embedding_path in SOURCE_EMBEDDING_PATHS:
        if not embedding_path.exists():
            LOGGER.error(MISSING_EMBEDDINGS_MESSAGE)
            return None

        LOGGER.info("Loading embeddings from %s", embedding_path)
        embedding_arrays.append(np.load(embedding_path))

    # Stack both years vertically so the autoencoder learns from one combined dataset.
    combined_embeddings = np.vstack(embedding_arrays).astype("float32")
    LOGGER.info("Loaded combined embeddings with shape %s", combined_embeddings.shape)
    return combined_embeddings


def validate_embedding_shape(embeddings: np.ndarray) -> None:
    """Validate that the loaded embeddings match the expected 384-dimensional shape."""
    if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIMENSION:
        raise ValueError(
            f"Expected embeddings with shape (N, {EMBEDDING_DIMENSION}), "
            f"got {embeddings.shape}."
        )


def create_dataloader(embeddings: np.ndarray) -> DataLoader:
    """Wrap NumPy embeddings in a PyTorch DataLoader for mini-batch training."""
    # Convert NumPy data into a torch tensor so PyTorch can move it onto GPU if available.
    tensor_embeddings = torch.from_numpy(embeddings)

    # TensorDataset makes each training example accessible by DataLoader in fixed-size batches.
    dataset = TensorDataset(tensor_embeddings)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def create_model() -> CVEAutoencoder:
    """Create a new autoencoder model and place it on the selected device."""
    model = CVEAutoencoder().to(DEVICE)
    LOGGER.info("Using device: %s", DEVICE)
    return model


def train_model(model: CVEAutoencoder, embeddings: np.ndarray) -> CVEAutoencoder:
    """Train the autoencoder to reconstruct the original embedding vectors."""
    dataloader = create_dataloader(embeddings)

    # MSE compares the reconstructed embedding against the original embedding.
    loss_function = nn.MSELoss()

    # Adam is a common optimizer that updates weights using gradient information.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for (batch_inputs,) in dataloader:
            # Move the current batch onto the chosen device before running the model.
            batch_inputs = batch_inputs.to(DEVICE)

            # Clear old gradients so PyTorch only uses gradients from this batch.
            optimizer.zero_grad()

            # Run the batch through the autoencoder and get reconstructed outputs.
            reconstructed = model(batch_inputs)

            # Compare the reconstructed vectors to the original input vectors.
            loss = loss_function(reconstructed, batch_inputs)
            loss.backward()
            optimizer.step()

            # Multiply by batch size so we can compute an average loss later.
            epoch_loss += loss.item() * batch_inputs.size(0)

        average_loss = epoch_loss / len(dataloader.dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {average_loss:.6f}")

    return model


def save_model(model: CVEAutoencoder) -> None:
    """Save the trained model weights to the models folder."""
    # Create the models folder automatically so saving works on a fresh setup.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    LOGGER.info("Saved autoencoder weights to %s", MODEL_PATH)


def load_model_weights(model: CVEAutoencoder) -> CVEAutoencoder | None:
    """Load saved model weights into a model instance for encoding mode."""
    if not MODEL_PATH.exists():
        LOGGER.error(MISSING_MODEL_MESSAGE)
        return None

    # map_location keeps loading safe even if the saved model was created on another device.
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    LOGGER.info("Loaded autoencoder weights from %s", MODEL_PATH)
    return model


def encode_embeddings(model: CVEAutoencoder, embeddings: np.ndarray) -> np.ndarray:
    """Run embeddings through the encoder only and return normalized 64-d vectors."""
    model.eval()

    # Convert the full embedding matrix into a tensor so the encoder can process it efficiently.
    input_tensor = torch.from_numpy(embeddings).to(DEVICE)

    with torch.no_grad():
        # Use only the encoder because we want compressed vectors, not reconstructions.
        compressed_tensor = model.encoder(input_tensor)

    # Move the encoded vectors back to CPU before converting them to a NumPy array.
    compressed_embeddings = compressed_tensor.cpu().numpy().astype("float32")

    # Normalize compressed vectors so FAISS can use cosine similarity later.
    faiss.normalize_L2(compressed_embeddings)
    return compressed_embeddings


def save_compressed_embeddings(compressed_embeddings: np.ndarray) -> None:
    """Save the compressed 64-dimensional vectors to the embeddings folder."""
    # Create the embeddings folder automatically so the compressed file can be written safely.
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(COMPRESSED_PATH, compressed_embeddings)
    LOGGER.info("Saved compressed embeddings to %s", COMPRESSED_PATH)


def run_training_workflow() -> None:
    """Train the autoencoder and then save compressed embeddings using the new model."""
    embeddings = load_combined_embeddings()
    if embeddings is None:
        return

    validate_embedding_shape(embeddings)
    model = create_model()
    trained_model = train_model(model, embeddings)
    save_model(trained_model)

    compressed_embeddings = encode_embeddings(trained_model, embeddings)
    save_compressed_embeddings(compressed_embeddings)
    print(f"Compressed embeddings shape: {compressed_embeddings.shape}")


def run_encoding_workflow() -> None:
    """Load an existing model and use it to save compressed embeddings only."""
    embeddings = load_combined_embeddings()
    if embeddings is None:
        return

    validate_embedding_shape(embeddings)
    model = create_model()
    loaded_model = load_model_weights(model)
    if loaded_model is None:
        return

    compressed_embeddings = encode_embeddings(loaded_model, embeddings)
    save_compressed_embeddings(compressed_embeddings)
    print(f"Compressed embeddings shape: {compressed_embeddings.shape}")


def main() -> None:
    """Run the autoencoder CLI in training mode or encoding-only mode."""
    configure_logging()
    args = parse_args()

    if args.train:
        run_training_workflow()
    elif args.encode:
        run_encoding_workflow()


if __name__ == "__main__":
    main()
