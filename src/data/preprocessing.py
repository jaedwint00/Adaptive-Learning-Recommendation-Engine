"""
Data preprocessing pipeline using Pandas, NumPy, and DuckDB
for the Adaptive Learning Recommendation Engine.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration for data preprocessing."""

    min_interactions: int = 5
    max_sequence_length: int = 100
    embedding_dim: int = 768
    test_split: float = 0.2
    validation_split: float = 0.1


class DataPreprocessor:
    """
    Advanced data preprocessing pipeline for recommendation systems.
    Handles user interactions, item features, and temporal patterns.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.conn = duckdb.connect()
        self.user_encoder: Dict[Any, int] = {}
        self.item_encoder: Dict[Any, int] = {}
        self.feature_stats: Dict[str, Any] = {}

    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from various formats (CSV, Parquet, JSON)."""
        path = Path(data_path)

        if path.suffix == ".csv":
            df = pd.read_csv(data_path)
        elif path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        elif path.suffix == ".json":
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        logger.info("Data loaded: %d rows, %d columns", df.shape[0], df.shape[1])
        logger.info("Data loaded from %s", data_path)
        return df

    def create_interaction_matrix(self, data: pd.DataFrame) -> Tuple[np.ndarray, Dict]:
        """Create user-item interaction matrix with DuckDB for efficiency."""

        # Use DuckDB for fast aggregations
        self.conn.execute("CREATE OR REPLACE TABLE interactions AS SELECT * FROM data")

        # Get user and item mappings
        users = self.conn.execute(
            """
            SELECT DISTINCT user_id
            FROM interactions
            GROUP BY user_id
            HAVING COUNT(*) >= ?
        """,
            [self.config.min_interactions],
        ).fetchdf()

        items = self.conn.execute(
            """
            SELECT DISTINCT item_id, COUNT(*) as frequency
            FROM interactions
            GROUP BY item_id
            ORDER BY frequency DESC
        """
        ).fetchdf()

        # Create interaction matrix using DuckDB for efficient processing
        query = """
        SELECT user_id, item_id, AVG(rating) as avg_rating,
               COUNT(*) as interaction_count
        FROM interactions
        GROUP BY user_id, item_id
        ORDER BY user_id, item_id
        """
        interactions = self.conn.execute(query).fetchdf()

        # Create encoders
        self.user_encoder = {user: idx for idx, user in enumerate(users["user_id"])}
        self.item_encoder = {item: idx for idx, item in enumerate(items["item_id"])}

        # Create interaction matrix
        n_users, n_items = len(self.user_encoder), len(self.item_encoder)
        interaction_matrix = np.zeros((n_users, n_items), dtype=np.float32)

        # Fill interaction matrix
        for _, row in interactions.iterrows():
            if (
                row["user_id"] in self.user_encoder
                and row["item_id"] in self.item_encoder
            ):
                user_idx = self.user_encoder[row["user_id"]]
                item_idx = self.item_encoder[row["item_id"]]
                interaction_matrix[user_idx, item_idx] = row.get("avg_rating", 1.0)

        metadata = {
            "n_users": n_users,
            "n_items": n_items,
            "density": np.count_nonzero(interaction_matrix) / (n_users * n_items),
            "user_encoder": self.user_encoder,
            "item_encoder": self.item_encoder,
        }

        return interaction_matrix, metadata

    def create_sequences(self, df: pd.DataFrame) -> Dict[str, List]:
        """Create sequential data for meta-learning."""

        # Sort by user and timestamp
        df_sorted = df.sort_values(["user_id", "timestamp"])

        sequences: Dict[str, List] = {
            "user_sequences": [],
            "item_sequences": [],
            "rating_sequences": [],
            "temporal_features": [],
        }

        for user_id in df_sorted["user_id"].unique():
            user_data = df_sorted[df_sorted["user_id"] == user_id]

            if len(user_data) < self.config.min_interactions:
                continue

            # Create sequences with sliding window
            items = user_data["item_id"].tolist()
            ratings = user_data.get("rating", [1.0] * len(items)).tolist()
            timestamps = pd.to_datetime(user_data["timestamp"]).tolist()

            # Truncate or pad sequences
            if len(items) > self.config.max_sequence_length:
                items = items[-self.config.max_sequence_length :]
                ratings = ratings[-self.config.max_sequence_length :]
                timestamps = timestamps[-self.config.max_sequence_length :]

            # Calculate temporal features
            time_diffs = []
            for i in range(1, len(timestamps)):
                diff = (
                    timestamps[i] - timestamps[i - 1]
                ).total_seconds() / 3600  # hours
                time_diffs.append(diff)
            time_diffs = [0] + time_diffs  # pad first element

            sequences["user_sequences"].append([user_id] * len(items))
            sequences["item_sequences"].append(items)
            sequences["rating_sequences"].append(ratings)
            sequences["temporal_features"].append(time_diffs)

        return sequences

    def extract_features(self, data: pd.DataFrame) -> Dict:
        """Extract and engineer features from the dataset."""
        features: Dict[str, Any] = {}

        # Use DuckDB for feature extraction
        self.conn.execute("CREATE OR REPLACE TABLE df AS SELECT * FROM data")
        # User features
        user_stats = self.conn.execute(
            """
            SELECT
                user_id,
                COUNT(*) as interaction_count,
                AVG(rating) as avg_rating,
                STDDEV(rating) as rating_std,
                COUNT(DISTINCT item_id) as n_unique_items,
                MAX(timestamp) - MIN(timestamp) as activity_span
            FROM df
            GROUP BY user_id
        """
        ).fetchdf()

        features["user_features"] = (
            user_stats.fillna(0).select_dtypes(include=[np.number]).values
        )

        # Item features
        item_stats = self.conn.execute(
            """
            SELECT
                item_id,
                COUNT(*) as popularity,
                AVG(rating) as avg_rating,
                STDDEV(rating) as rating_std,
                COUNT(DISTINCT user_id) as n_unique_users
            FROM df
            GROUP BY item_id
        """
        ).fetchdf()

        features["item_features"] = (
            item_stats.fillna(0).select_dtypes(include=[np.number]).values
        )

        # Contextual features (time-based)
        data["hour"] = pd.to_datetime(data["timestamp"]).dt.hour
        data["day_of_week"] = pd.to_datetime(data["timestamp"]).dt.dayofweek
        data["month"] = pd.to_datetime(data["timestamp"]).dt.month

        temporal_features = data[["hour", "day_of_week", "month"]].values
        features["temporal_features"] = temporal_features

        # Store feature statistics for normalization
        self.feature_stats = {
            "user_mean": np.mean(features["user_features"], axis=0),
            "user_std": np.std(features["user_features"], axis=0),
            "item_mean": np.mean(features["item_features"], axis=0),
            "item_std": np.std(features["item_features"], axis=0),
        }

        return features

    def normalize_features(
        self, features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Normalize features using stored statistics."""

        normalized = {}

        # Normalize user features
        normalized["user_features"] = (
            features["user_features"] - self.feature_stats["user_mean"]
        ) / (self.feature_stats["user_std"] + 1e-8)

        # Normalize item features
        normalized["item_features"] = (
            features["item_features"] - self.feature_stats["item_mean"]
        ) / (self.feature_stats["item_std"] + 1e-8)

        # Temporal features are already normalized (0-23 for hour, etc.)
        normalized["temporal_features"] = features["temporal_features"]

        return normalized

    def create_meta_learning_tasks(
        self, sequences: Dict, n_tasks: int = 1000
    ) -> List[Dict]:
        """Create meta-learning tasks for few-shot recommendation."""

        tasks = []

        for _ in range(n_tasks):
            # Sample a random user sequence
            user_idx = np.random.randint(len(sequences["user_sequences"]))

            user_seq = sequences["user_sequences"][user_idx]
            item_seq = sequences["item_sequences"][user_idx]
            rating_seq = sequences["rating_sequences"][user_idx]
            temporal_seq = sequences["temporal_features"][user_idx]

            if (
                len(item_seq) < 4
            ):  # Need at least 4 interactions for support/query split
                continue

            # Split into support and query sets
            split_point = len(item_seq) // 2

            task = {
                "support": {
                    "users": user_seq[:split_point],
                    "items": item_seq[:split_point],
                    "ratings": rating_seq[:split_point],
                    "temporal": temporal_seq[:split_point],
                },
                "query": {
                    "users": user_seq[split_point:],
                    "items": item_seq[split_point:],
                    "ratings": rating_seq[split_point:],
                    "temporal": temporal_seq[split_point:],
                },
                "user_id": user_seq[0],  # All elements should be the same user
            }

            tasks.append(task)

        return tasks

    def split_data(
        self,
        data: Dict,
        test_size: Optional[float] = None,
        val_size: Optional[float] = None,
    ) -> Dict:
        """Split data into train/validation/test sets."""

        if test_size is None:
            test_size = self.config.test_split
        if val_size is None:
            val_size = self.config.validation_split

        n_samples = len(data[list(data.keys())[0]])
        indices = np.random.permutation(n_samples)

        test_split = int(n_samples * test_size)
        val_split = int(n_samples * val_size)

        train_indices = indices[test_split + val_split :]
        val_indices = indices[test_split : test_split + val_split]
        test_indices = indices[:test_split]

        splits: Dict[str, Dict] = {}
        for split_name, split_indices in [
            ("train", train_indices),
            ("val", val_indices),
            ("test", test_indices),
        ]:
            splits[split_name] = {}
            for key, values in data.items():
                if isinstance(values, np.ndarray):
                    splits[split_name][key] = values[split_indices]
                elif isinstance(values, list):
                    splits[split_name][key] = [values[i] for i in split_indices]

        return splits

    def save_processed_data(self, data: Dict, output_path: str) -> None:
        """Save processed data to disk."""

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save numpy arrays
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np.save(output_dir / f"{key}.npy", value)
            elif isinstance(value, dict):
                # Save nested dictionaries recursively
                nested_path = output_dir / key
                nested_path.mkdir(exist_ok=True)
                self.save_processed_data(value, str(nested_path))

        # Save encoders and metadata
        import pickle  # pylint: disable=import-outside-toplevel

        with open(output_dir / "encoders.pkl", "wb") as f:
            pickle.dump(
                {
                    "user_encoder": self.user_encoder,
                    "item_encoder": self.item_encoder,
                    "feature_stats": self.feature_stats,
                },
                f,
            )

        logger.info("Processed data saved to %s", output_path)


def main():
    """Example usage of the data preprocessing pipeline."""

    # Configuration
    config = DataConfig(
        min_interactions=10,
        max_sequence_length=50,
        test_split=0.2,
        validation_split=0.1,
    )

    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)

    # Example: Create sample data for testing
    np.random.seed(42)
    n_users, n_items = 1000, 500
    n_interactions = 10000

    sample_data = pd.DataFrame(
        {
            "user_id": np.random.randint(0, n_users, n_interactions),
            "item_id": np.random.randint(0, n_items, n_interactions),
            "rating": np.random.uniform(1, 5, n_interactions),
            "timestamp": pd.date_range("2023-01-01", periods=n_interactions, freq="1H"),
        }
    )

    print("Sample data created:")
    print(sample_data.head())

    # Process data
    interaction_matrix, metadata = preprocessor.create_interaction_matrix(sample_data)
    sequences = preprocessor.create_sequences(sample_data)
    features = preprocessor.extract_features(sample_data)
    normalized_features = preprocessor.normalize_features(features)
    meta_tasks = preprocessor.create_meta_learning_tasks(sequences, n_tasks=100)

    print("\nProcessing complete:")
    print(f"Interaction matrix shape: {interaction_matrix.shape}")
    print(f"Matrix density: {metadata['density']:.4f}")
    print(f"Number of sequences: {len(sequences['user_sequences'])}")
    print(f"Number of meta-learning tasks: {len(meta_tasks)}")

    # Save processed data
    processed_data = {
        "interaction_matrix": interaction_matrix,
        "sequences": sequences,
        "features": normalized_features,
        "meta_tasks": meta_tasks,
        "metadata": metadata,
    }

    preprocessor.save_processed_data(processed_data, "data/processed")


if __name__ == "__main__":
    main()
