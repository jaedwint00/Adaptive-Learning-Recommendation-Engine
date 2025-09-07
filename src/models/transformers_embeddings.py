"""
Transformer-based embeddings and semantic similarity for recommendation systems.
Uses Hugging Face Transformers for content understanding and user preference modeling.
"""

import logging
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import faiss
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for transformer embeddings."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 512
    batch_size: int = 32
    embedding_dim: int = 384
    cache_embeddings: bool = True
    use_gpu: bool = True
    pooling_strategy: str = "mean"  # mean, cls, max


class TransformerEmbeddingModel(nn.Module):
    """
    Transformer-based embedding model for items and user preferences.
    Supports multiple pre-trained models and pooling strategies.
    """

    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

        # Load pre-trained model and tokenizer
        if "sentence-transformers" in config.model_name:
            self.model = SentenceTransformer(config.model_name)
            self.tokenizer = None
            self.is_sentence_transformer = True
        else:
            self.model = AutoModel.from_pretrained(config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            self.is_sentence_transformer = False

        self.model.to(self.device)
        self.embedding_cache: Optional[Dict[str, torch.Tensor]] = (
            {} if config.cache_embeddings else None
        )

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of texts into embeddings."""

        if self.is_sentence_transformer:
            return self._encode_with_sentence_transformer(texts)
        return self._encode_with_transformer(texts)

    def _encode_with_sentence_transformer(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using SentenceTransformer."""

        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
        )

        return embeddings

    def _encode_with_transformer(self, texts: List[str]) -> torch.Tensor:
        """Encode texts using standard transformer model."""

        embeddings = []

        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i : i + self.config.batch_size]

            # Tokenize
            if self.tokenizer is None:
                raise ValueError("Tokenizer not initialized")
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            ).to(self.device)

            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)

                if self.config.pooling_strategy == "cls":
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]
                elif self.config.pooling_strategy == "mean":
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = (
                        attention_mask.unsqueeze(-1)
                        .expand(token_embeddings.size())
                        .float()
                    )
                    batch_embeddings = torch.sum(
                        token_embeddings * input_mask_expanded, 1
                    ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                elif self.config.pooling_strategy == "max":
                    token_embeddings = outputs.last_hidden_state
                    batch_embeddings = torch.max(token_embeddings, dim=1)[0]
                else:
                    raise ValueError(
                        f"Unknown pooling strategy: {self.config.pooling_strategy}"
                    )

            embeddings.append(batch_embeddings)

        return torch.cat(embeddings, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the embedding model."""
        # This method is required by nn.Module but not used in our implementation
        # since we use encode_texts for text encoding
        return x

    def compute_similarity(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity between two sets of embeddings."""

        # Normalize embeddings
        embeddings1_norm = F.normalize(embeddings1, p=2, dim=1)
        embeddings2_norm = F.normalize(embeddings2, p=2, dim=1)

        # Compute cosine similarity
        similarity = torch.mm(embeddings1_norm, embeddings2_norm.t())

        return similarity


class ContentBasedRecommender:
    """
    Content-based recommender using transformer embeddings.
    Recommends items based on semantic similarity to user preferences.
    """

    def __init__(self, embedding_model: TransformerEmbeddingModel):
        self.embedding_model = embedding_model
        self.item_embeddings: Optional[torch.Tensor] = None
        self.item_index: Optional[Dict[int, str]] = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.item_metadata: Dict[str, Any] = {}

    def fit_items(self, items: Dict[str, str]):
        """
        Fit the recommender on item descriptions.

        Args:
            items: Dictionary mapping item_id to item description/content
        """

        item_ids = list(items.keys())
        item_texts = list(items.values())

        logger.info("Encoding %d item descriptions...", len(item_texts))

        # Encode item descriptions
        self.item_embeddings = self.embedding_model.encode_texts(item_texts)
        self.item_index = dict(enumerate(item_ids))

        # Build FAISS index for fast similarity search
        self._build_faiss_index()

        logger.info("Item embeddings computed and indexed")

    def _build_faiss_index(self):
        """Build FAISS index for efficient similarity search."""
        if self.item_embeddings is None:
            raise ValueError("Item embeddings not computed yet")

        embeddings_np = self.item_embeddings.cpu().numpy().astype("float32")

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_np)  # pylint: disable=no-value-for-parameter

        # Create FAISS index
        dimension = embeddings_np.shape[1]
        # Inner product for cosine similarity
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings_np)  # pylint: disable=no-value-for-parameter

        logger.info("FAISS index built with %d items", self.faiss_index.ntotal)

    def get_user_profile_embedding(self, user_interactions: List[Dict]) -> torch.Tensor:
        """
        Create user profile embedding from interaction history.

        Args:
            user_interactions: List of user interactions with 'item_id', 'rating', etc.
        """

        # Get embeddings for interacted items
        interacted_items = []
        weights = []

        for interaction in user_interactions:
            item_id = interaction["item_id"]
            rating = interaction.get("rating", 1.0)

            # Find item embedding
            if self.item_index is None or self.item_embeddings is None:
                continue

            item_idx = None
            for idx, stored_item_id in self.item_index.items():
                if stored_item_id == item_id:
                    item_idx = idx
                    break

            if item_idx is not None:
                interacted_items.append(self.item_embeddings[item_idx])
                weights.append(rating)

        if not interacted_items:
            # Return zero embedding if no interactions found
            return torch.zeros(
                self.embedding_model.config.embedding_dim,
                device=self.embedding_model.device,
            )

        # Weighted average of item embeddings
        interacted_embeddings = torch.stack(interacted_items)
        weights_tensor = torch.tensor(weights, device=self.embedding_model.device)
        weights_tensor = weights_tensor / weights_tensor.sum()  # Normalize weights

        user_profile = torch.sum(
            interacted_embeddings * weights_tensor.unsqueeze(1), dim=0
        )

        return user_profile

    def recommend_items(
        self,
        user_profile: torch.Tensor,
        k: int = 10,
        exclude_items: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Recommend top-k items for a user based on their profile embedding.

        Args:
            user_profile: User profile embedding
            k: Number of recommendations
            exclude_items: List of item IDs to exclude from recommendations

        Returns:
            List of (item_id, similarity_score) tuples
        """

        if self.faiss_index is None:
            raise ValueError("Model not fitted. Call fit_items() first.")

        # Normalize user profile
        user_profile_np = (
            F.normalize(user_profile.unsqueeze(0), p=2, dim=1)
            .cpu()
            .numpy()
            .astype("float32")
        )

        # Search for similar items
        # pylint: disable=no-value-for-parameter
        similarities, indices = self.faiss_index.search(
            user_profile_np, k * 2
        )  # Get more than needed for filtering

        recommendations = []
        exclude_set = set(exclude_items) if exclude_items else set()

        for sim, idx in zip(similarities[0], indices[0]):
            if self.item_index is None:
                break
            item_id = self.item_index[idx]

            if item_id not in exclude_set:
                recommendations.append((item_id, float(sim)))

            if len(recommendations) >= k:
                break

        return recommendations

    def explain_recommendation(
        self, user_profile: torch.Tensor, item_id: str, user_interactions: List[Dict]
    ) -> Dict[str, Any]:
        """
        Provide explanation for why an item was recommended.

        Args:
            user_profile: User profile embedding
            item_id: Recommended item ID
            user_interactions: User's interaction history

        Returns:
            Dictionary with explanation details
        """

        # Find item embedding
        if self.item_index is None or self.item_embeddings is None:
            return {"error": "Model not fitted"}

        item_idx = None
        for idx, stored_item_id in self.item_index.items():
            if stored_item_id == item_id:
                item_idx = idx
                break

        if item_idx is None:
            return {"error": "Item not found"}

        item_embedding = self.item_embeddings[item_idx]

        # Compute similarity with user profile
        overall_similarity = torch.cosine_similarity(
            user_profile.unsqueeze(0), item_embedding.unsqueeze(0)
        ).item()

        # Find most similar items from user history
        similar_items = []
        for interaction in user_interactions:
            hist_item_id = interaction["item_id"]
            hist_item_idx = None

            if self.item_index is not None and self.item_embeddings is not None:
                for idx, stored_item_id in self.item_index.items():
                    if stored_item_id == hist_item_id:
                        hist_item_idx = idx
                        break

                if hist_item_idx is not None:
                    hist_embedding = self.item_embeddings[hist_item_idx]
                    similarity = torch.cosine_similarity(
                        item_embedding.unsqueeze(0), hist_embedding.unsqueeze(0)
                    ).item()
                    similar_items.append(
                        {
                            "item_id": hist_item_id,
                            "similarity": similarity,
                            "rating": interaction.get("rating", 1.0),
                        }
                    )

        # Sort by similarity
        similar_items.sort(key=lambda x: x["similarity"], reverse=True)

        explanation = {
            "recommended_item": item_id,
            "overall_similarity": overall_similarity,
            "most_similar_items": similar_items[:5],
            "explanation": (
                f"This item is recommended because it's similar to items "
                f"you've interacted with, with an overall similarity "
                f"score of {overall_similarity:.3f}"
            ),
        }

        return explanation


class HybridTransformerRecommender:
    """
    Hybrid recommender combining collaborative filtering with content-based filtering
    using transformer embeddings.
    """

    def __init__(
        self,
        embedding_model: TransformerEmbeddingModel,
        cf_weight: float = 0.6,
        content_weight: float = 0.4,
    ):
        self.embedding_model = embedding_model
        self.content_recommender = ContentBasedRecommender(embedding_model)
        self.cf_weight = cf_weight
        self.content_weight = content_weight

        # Collaborative filtering components
        self.user_item_matrix: Optional[np.ndarray] = None
        self.user_similarities: Optional[np.ndarray] = None
        self.item_similarities: Optional[np.ndarray] = None
        self.user_encoder: Optional[Dict[str, int]] = None
        self.item_encoder: Optional[Dict[str, int]] = None

    def fit(
        self,
        interaction_matrix: np.ndarray,
        items: Dict[str, str],
        user_encoder: Dict[str, int],
        item_encoder: Dict[str, int],
    ):
        """
        Fit the hybrid recommender.

        Args:
            interaction_matrix: User-item interaction matrix
            items: Dictionary mapping item_id to item description
            user_encoder: Mapping from user_id to matrix index
            item_encoder: Mapping from item_id to matrix index
        """

        # Fit content-based component
        self.content_recommender.fit_items(items)

        # Fit collaborative filtering component
        self.user_item_matrix = interaction_matrix
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder

        # Compute user and item similarities for CF
        self._compute_cf_similarities()

        logger.info("Hybrid recommender fitted successfully")

    def _compute_cf_similarities(self):
        """Compute user and item similarities for collaborative filtering."""
        if self.user_item_matrix is None:
            raise ValueError("User-item matrix not set")

        # User similarities (cosine similarity between user vectors)
        self.user_similarities = cosine_similarity(self.user_item_matrix)

        # Item similarities (cosine similarity between item vectors)
        self.item_similarities = cosine_similarity(self.user_item_matrix.T)

        logger.info("CF similarities computed")

    def get_cf_recommendations(
        self, user_idx: int, k: int = 10
    ) -> List[Tuple[int, float]]:
        """Get collaborative filtering recommendations."""

        if self.user_similarities is None or self.user_item_matrix is None:
            return []

        # Find similar users
        user_sim = self.user_similarities[user_idx]
        # Top 10 similar users (excluding self)
        similar_users = np.argsort(user_sim)[::-1][1:11]

        # Get items liked by similar users
        user_ratings = self.user_item_matrix[user_idx]
        recommendations = {}

        for similar_user in similar_users:
            sim_score = user_sim[similar_user]
            similar_user_ratings = self.user_item_matrix[similar_user]

            for item_idx, rating in enumerate(similar_user_ratings):
                # Item not rated by target user
                if rating > 0 and user_ratings[item_idx] == 0:
                    if item_idx not in recommendations:
                        recommendations[item_idx] = 0
                    recommendations[item_idx] += sim_score * rating

        # Sort and return top-k
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        return [(int(item_idx), float(score)) for item_idx, score in sorted_recs[:k]]

    def recommend(
        self, user_id: str, user_interactions: List[Dict], k: int = 10
    ) -> List[Dict]:
        """
        Generate hybrid recommendations combining CF and content-based approaches.

        Args:
            user_id: User identifier
            user_interactions: User's interaction history
            k: Number of recommendations

        Returns:
            List of recommendation dictionaries with scores and explanations
        """

        recommendations = {}

        # Content-based recommendations
        user_profile = self.content_recommender.get_user_profile_embedding(
            user_interactions
        )
        content_recs = self.content_recommender.recommend_items(
            user_profile,
            k=k * 2,
            exclude_items=[interaction["item_id"] for interaction in user_interactions],
        )

        for item_id, score in content_recs:
            recommendations[item_id] = {
                "content_score": score,
                "cf_score": 0.0,
                "hybrid_score": 0.0,
            }

        # Collaborative filtering recommendations
        if (
            self.user_encoder is not None
            and self.item_encoder is not None
            and user_id in self.user_encoder
        ):
            user_idx = self.user_encoder[user_id]
            cf_recs = self.get_cf_recommendations(user_idx, k=k * 2)

            # Convert item indices back to item IDs
            reverse_item_encoder = {v: k for k, v in self.item_encoder.items()}

            for item_idx, score in cf_recs:
                if item_idx in reverse_item_encoder:
                    item_id = reverse_item_encoder[item_idx]

                    if item_id not in recommendations:
                        recommendations[item_id] = {
                            "content_score": 0.0,
                            "cf_score": score,
                            "hybrid_score": 0.0,
                        }
                    else:
                        recommendations[item_id]["cf_score"] = score

        # Compute hybrid scores
        for item_id, scores in recommendations.items():
            content_score = scores["content_score"]
            cf_score = scores["cf_score"]

            # Normalize scores to [0, 1] range
            # Cosine similarity is in [-1, 1]
            content_score_norm = (content_score + 1) / 2
            max_cf_score = max(
                (r["cf_score"] for r in recommendations.values()), default=1.0
            )
            cf_score_norm = cf_score / max(1.0, max_cf_score)

            hybrid_score = (
                self.content_weight * content_score_norm
                + self.cf_weight * cf_score_norm
            )

            scores["hybrid_score"] = hybrid_score

        # Sort by hybrid score and return top-k
        sorted_recs = sorted(
            recommendations.items(), key=lambda x: x[1]["hybrid_score"], reverse=True
        )

        final_recommendations = []
        for item_id, scores in sorted_recs[:k]:
            rec = {
                "item_id": item_id,
                "hybrid_score": scores["hybrid_score"],
                "content_score": scores["content_score"],
                "cf_score": scores["cf_score"],
                "explanation": self._generate_explanation(scores),
            }
            final_recommendations.append(rec)

        return final_recommendations

    def _generate_explanation(self, scores: Dict[str, float]) -> str:
        """Generate explanation for recommendation."""

        content_contrib = self.content_weight * scores["content_score"]
        cf_contrib = self.cf_weight * scores["cf_score"]

        if content_contrib > cf_contrib:
            return (
                f"Recommended based on content similarity "
                f"(score: {scores['content_score']:.3f})"
            )
        return (
            f"Recommended based on similar users' preferences "
            f"(score: {scores['cf_score']:.3f})"
        )

    def save_model(self, path: str):
        """Save the hybrid model to disk."""

        model_data = {
            "user_item_matrix": self.user_item_matrix,
            "user_similarities": self.user_similarities,
            "item_similarities": self.item_similarities,
            "user_encoder": self.user_encoder,
            "item_encoder": self.item_encoder,
            "cf_weight": self.cf_weight,
            "content_weight": self.content_weight,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info("Hybrid model saved to %s", path)

    def load_model(self, path: str):
        """Load the hybrid model from disk."""

        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.user_item_matrix = model_data["user_item_matrix"]
        self.user_similarities = model_data["user_similarities"]
        self.item_similarities = model_data["item_similarities"]
        self.user_encoder = model_data["user_encoder"]
        self.item_encoder = model_data["item_encoder"]
        self.cf_weight = model_data["cf_weight"]
        self.content_weight = model_data["content_weight"]

        logger.info("Hybrid model loaded from %s", path)


def main():
    """Example usage of transformer-based recommendation system."""

    # Configuration
    config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_length=256,
        batch_size=16,
        use_gpu=torch.cuda.is_available(),
    )

    # Initialize embedding model
    embedding_model = TransformerEmbeddingModel(config)

    # Sample data
    items = {
        "item_1": "Action movie with explosions and car chases",
        "item_2": "Romantic comedy about finding love in the city",
        "item_3": "Science fiction thriller set in space",
        "item_4": "Documentary about climate change",
        "item_5": "Horror movie with supernatural elements",
    }

    user_interactions = [
        {"item_id": "item_1", "rating": 4.5},
        {"item_id": "item_3", "rating": 4.0},
    ]

    # Content-based recommendations
    print("Testing content-based recommendations...")
    content_recommender = ContentBasedRecommender(embedding_model)
    content_recommender.fit_items(items)

    user_profile = content_recommender.get_user_profile_embedding(user_interactions)
    recommendations = content_recommender.recommend_items(
        user_profile, k=3, exclude_items=["item_1", "item_3"]
    )

    print("Content-based recommendations:")
    for item_id, score in recommendations:
        print(f"  {item_id}: {score:.4f}")

    # Explanation
    if recommendations:
        explanation = content_recommender.explain_recommendation(
            user_profile, recommendations[0][0], user_interactions
        )
        print(f"\nExplanation for {recommendations[0][0]}:")
        print(f"  {explanation['explanation']}")

    print("\nTransformer-based recommendation system demo complete!")


if __name__ == "__main__":
    main()
