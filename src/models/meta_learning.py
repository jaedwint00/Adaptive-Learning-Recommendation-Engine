"""
Meta-learning algorithms for adaptive recommendation systems.
Implements MAML (Model-Agnostic Meta-Learning) and Prototypical Networks.
"""

import copy
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class MetaLearningDataset(Dataset):
    """Dataset for meta-learning tasks."""

    def __init__(self, tasks: List[Dict], n_items: int = 1000, support_size: int = 5):
        self.tasks = tasks
        self.n_items = n_items
        self.support_size = support_size

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]

        # Convert to tensors
        support_users = torch.tensor(task["support"]["users"], dtype=torch.long)
        support_items = torch.randint(0, self.n_items, (self.support_size,))
        support_ratings = torch.tensor(task["support"]["ratings"], dtype=torch.float32)

        query_users = torch.tensor(task["query"]["users"], dtype=torch.long)
        query_items = torch.tensor(task["query"]["items"], dtype=torch.long)
        query_ratings = torch.tensor(task["query"]["ratings"], dtype=torch.float32)

        return {
            "support": {
                "users": support_users,
                "items": support_items,
                "ratings": support_ratings,
            },
            "query": {
                "users": query_users,
                "items": query_items,
                "ratings": query_ratings,
            },
            "user_id": task["user_id"],
        }


class BaseRecommendationModel(nn.Module, ABC):
    """Base class for recommendation models."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,  # pylint: disable=unused-argument
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)

        # Initialize embeddings
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    @abstractmethod
    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Forward pass for the recommendation model.
        
        Args:
            users: User tensor indices
            items: Item tensor indices
            
        Returns:
            Predicted ratings tensor
        """


class MatrixFactorizationModel(BaseRecommendationModel):
    """Simple matrix factorization model for meta-learning."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__(n_users, n_items, embedding_dim)

        # Set default hidden dimensions if None provided
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # MLP layers for interaction modeling
        layers = []
        input_dim = embedding_dim * 2

        for hidden_dim in hidden_dims:
            layers.extend(
                [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
            )
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embeddings(users)
        item_emb = self.item_embeddings(items)

        # Concatenate embeddings
        interaction = torch.cat([user_emb, item_emb], dim=-1)

        # Pass through MLP
        output = self.mlp(interaction)
        return output.squeeze(-1)


class MAML:
    """
    Model-Agnostic Meta-Learning for recommendation systems.
    Enables fast adaptation to new users with few interactions.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 5,
    ):
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.inner_steps = inner_steps
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=meta_lr)

    def inner_update(self, model: nn.Module, support_data: Dict) -> nn.Module:
        """Perform inner loop update on support set."""

        # Create a copy of the model for inner updates
        adapted_model = copy.deepcopy(model)
        inner_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        users = support_data["users"]
        items = support_data["items"]
        ratings = support_data["ratings"]

        for _ in range(self.inner_steps):
            inner_optimizer.zero_grad()

            predictions = adapted_model(users, items)
            loss = F.mse_loss(predictions, ratings)

            loss.backward()
            inner_optimizer.step()

        return adapted_model

    def meta_update(self, batch: List[Dict]) -> float:
        """Perform meta-update across a batch of tasks."""

        meta_loss = torch.tensor(0.0, requires_grad=True)

        for task in batch:
            support_data = task["support"]
            query_data = task["query"]

            # Inner loop: adapt to support set
            adapted_model = self.inner_update(self.model, support_data)

            # Outer loop: evaluate on query set
            query_predictions = adapted_model(query_data["users"], query_data["items"])
            query_loss = F.mse_loss(query_predictions, query_data["ratings"])

            meta_loss = meta_loss + query_loss

        # Average across tasks
        meta_loss = meta_loss / len(batch)

        # Meta-gradient update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()

    def adapt_to_user(
        self, user_data: Dict, n_steps: Optional[int] = None
    ) -> nn.Module:
        """Adapt the model to a specific user's preferences."""

        if n_steps is None:
            n_steps = self.inner_steps

        adapted_model = copy.deepcopy(self.model)
        optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)

        users = user_data["users"]
        items = user_data["items"]
        ratings = user_data["ratings"]

        for _ in range(n_steps):
            optimizer.zero_grad()

            predictions = adapted_model(users, items)
            loss = F.mse_loss(predictions, ratings)

            loss.backward()
            optimizer.step()

        return adapted_model

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""

        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            loss = self.meta_update(batch)
            total_loss += loss
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for few-shot recommendation.
    Learns to create prototypes for different user preferences.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        prototype_dim: int = 128,
    ):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.prototype_dim = prototype_dim

        # Embedding layers
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)

        # Encoder network to create prototypes
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, prototype_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(prototype_dim, prototype_dim),
            nn.ReLU(),
            nn.Linear(prototype_dim, prototype_dim),
        )

        # Initialize embeddings
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def encode_interaction(
        self, users: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        """Encode user-item interactions into prototype space."""

        user_emb = self.user_embeddings(users)
        item_emb = self.item_embeddings(items)

        # Concatenate embeddings
        interaction = torch.cat([user_emb, item_emb], dim=-1)

        # Encode to prototype space
        encoded = self.encoder(interaction)
        return encoded

    def create_prototypes(self, support_data: Dict) -> torch.Tensor:
        """Create prototypes from support set."""

        users = support_data["users"]
        items = support_data["items"]
        ratings = support_data["ratings"]

        # Encode interactions
        encoded = self.encode_interaction(users, items)

        # Create prototypes based on rating levels
        # For simplicity, we'll create prototypes for high (>3.5) and low (<=3.5)
        # ratings
        high_mask = ratings > 3.5
        low_mask = ratings <= 3.5

        prototypes = []

        if high_mask.sum() > 0:
            high_prototype = encoded[high_mask].mean(dim=0)
            prototypes.append(high_prototype)

        if low_mask.sum() > 0:
            low_prototype = encoded[low_mask].mean(dim=0)
            prototypes.append(low_prototype)

        if len(prototypes) == 0:
            # Fallback: use overall mean
            prototypes.append(encoded.mean(dim=0))

        return torch.stack(prototypes)

    def predict_with_prototypes(
        self, query_data: Dict, prototypes: torch.Tensor
    ) -> torch.Tensor:
        """Make predictions using prototypes."""

        users = query_data["users"]
        items = query_data["items"]

        # Encode query interactions
        query_encoded = self.encode_interaction(users, items)

        # Calculate distances to prototypes
        distances = torch.cdist(
            query_encoded.unsqueeze(0), prototypes.unsqueeze(0)
        ).squeeze(0)

        # Use negative distance as similarity score
        similarities = -distances

        # Weighted combination of prototype similarities
        # For now, simple average (can be made more sophisticated)
        predictions = similarities.mean(dim=-1)

        # Scale to rating range [1, 5]
        predictions = torch.sigmoid(predictions) * 4 + 1

        return predictions

    def forward(self, support_data: Dict, query_data: Dict) -> torch.Tensor:
        """Forward pass for training."""

        prototypes = self.create_prototypes(support_data)
        predictions = self.predict_with_prototypes(query_data, prototypes)

        return predictions


class MetaLearningTrainer:
    """Trainer for meta-learning models."""

    def __init__(self, model_type: str = "maml", **model_kwargs):
        self.model_type = model_type
        self.model_kwargs = model_kwargs
        self.model: Optional[nn.Module] = None
        self.trainer: Optional[Union[MAML, PrototypicalNetwork]] = None

    def build_model(self, n_users: int, n_items: int):
        """Build the meta-learning model."""

        if self.model_type == "maml":
            base_model = MatrixFactorizationModel(n_users, n_items, **self.model_kwargs)
            self.trainer = MAML(base_model)
            self.model = base_model
        elif self.model_type == "prototypical":
            self.trainer = PrototypicalNetwork(n_users, n_items, **self.model_kwargs)
            self.model = self.trainer
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        train_tasks: List[Dict],
        val_tasks: Optional[List[Dict]] = None,
        epochs: int = 100,
        batch_size: int = 32,
    ):
        """Train the meta-learning model."""

        # Create datasets
        train_dataset = MetaLearningDataset(train_tasks)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if val_tasks:
            val_dataset = MetaLearningDataset(val_tasks)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Training
            if self.model_type == "maml" and self.trainer is not None:
                train_loss = self.trainer.train_epoch(train_loader)
            else:
                train_loss = self._train_prototypical_epoch(train_loader)

            # Validation
            if val_tasks:
                val_loss = self._validate(val_loader)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint(epoch)

                logger.info(
                    "Epoch %d/%d - Train Loss: %.4f, Val Loss: %.4f",
                    epoch + 1,
                    epochs,
                    train_loss,
                    val_loss,
                )
            else:
                logger.info(
                    "Epoch %d/%d - Train Loss: %.4f", epoch + 1, epochs, train_loss
                )

    def _train_prototypical_epoch(self, dataloader: DataLoader) -> float:
        """Train prototypical network for one epoch."""

        if self.trainer is not None and hasattr(self.trainer, "train"):
            self.trainer.train()
        if self.trainer is not None and hasattr(self.trainer, "parameters"):
            optimizer = optim.Adam(self.trainer.parameters(), lr=0.001)
        else:
            raise ValueError("Trainer not initialized")

        total_loss = torch.tensor(0.0, requires_grad=True)
        num_batches = 0

        for batch in dataloader:
            optimizer.zero_grad()

            batch_loss = torch.tensor(0.0, requires_grad=True)
            for task in batch:
                support_data = task["support"]
                query_data = task["query"]

                if self.model is not None:
                    predictions = self.model(support_data, query_data)
                    loss = F.mse_loss(predictions, query_data["ratings"])
                    batch_loss = batch_loss + loss

            batch_loss = batch_loss / len(batch)
            batch_loss.backward()
            optimizer.step()

            total_loss = total_loss + batch_loss.detach()
            num_batches += 1

        return (total_loss / num_batches).item() if num_batches > 0 else 0.0

    def _validate(self, dataloader: DataLoader) -> float:
        """Validate the model."""

        if self.model is not None:
            self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch_loss = 0.0

                for task in batch:
                    support_data = task["support"]
                    query_data = task["query"]

                    if self.model_type == "maml" and self.trainer is not None:
                        adapted_model = self.trainer.adapt_to_user(support_data)
                        predictions = adapted_model(
                            query_data["users"], query_data["items"]
                        )
                    elif self.model is not None:
                        predictions = self.model(support_data, query_data)
                    else:
                        continue

                    loss = F.mse_loss(predictions, query_data["ratings"])
                    batch_loss += loss.item()

                batch_loss = batch_loss / len(batch)
                total_loss += batch_loss
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""

        if self.model is not None:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "model_type": self.model_type,
                "model_kwargs": self.model_kwargs,
            }

            torch.save(checkpoint, f"checkpoints/meta_model_epoch_{epoch}.pt")
            logger.info("Checkpoint saved at epoch %d", epoch)

    def adapt_to_user(self, user_data: Dict) -> nn.Module:
        """Adapt model to user preferences."""

        if self.trainer is not None:
            return self.trainer.adapt_to_user(user_data)
        if self.model is not None:
            return self.model
        raise ValueError("Neither trainer nor model is initialized")


def main():
    """Example usage of meta-learning models."""

    # Create sample data
    n_users, n_items = 1000, 500
    n_tasks = 1000

    # Generate sample meta-learning tasks
    tasks = []
    for _ in range(n_tasks):
        n_support = np.random.randint(3, 10)
        n_query = np.random.randint(2, 8)

        user_id = np.random.randint(0, n_users)

        task = {
            "support": {
                "users": [user_id] * n_support,
                "items": np.random.randint(0, n_items, n_support).tolist(),
                "ratings": np.random.uniform(1, 5, n_support).tolist(),
            },
            "query": {
                "users": [user_id] * n_query,
                "items": np.random.randint(0, n_items, n_query).tolist(),
                "ratings": np.random.uniform(1, 5, n_query).tolist(),
            },
            "user_id": user_id,
        }
        tasks.append(task)

    # Split tasks
    split_idx = int(0.8 * len(tasks))
    train_tasks = tasks[:split_idx]
    val_tasks = tasks[split_idx:]

    print(
        f"Created {len(train_tasks)} training tasks and {len(val_tasks)} validation tasks"
    )

    # Train MAML model
    print("\nTraining MAML model...")
    maml_trainer = MetaLearningTrainer(
        model_type="maml", embedding_dim=64, hidden_dims=[128, 64]
    )
    maml_trainer.build_model(n_users, n_items)
    maml_trainer.train(train_tasks, val_tasks, epochs=10, batch_size=16)

    # Train Prototypical Network
    print("\nTraining Prototypical Network...")
    proto_trainer = MetaLearningTrainer(
        model_type="prototypical", embedding_dim=64, prototype_dim=128
    )
    proto_trainer.build_model(n_users, n_items)
    proto_trainer.train(train_tasks, val_tasks, epochs=10, batch_size=16)

    print("Meta-learning training complete!")


if __name__ == "__main__":
    main()
