"""
FastAPI application for the Adaptive Learning Recommendation Engine.
Provides RESTful endpoints for recommendation serving with async inference.
"""

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import List, Dict, Optional, Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..models.meta_learning import MetaLearningTrainer
from ..models.transformers_embeddings import (
    TransformerEmbeddingModel,
    HybridTransformerRecommender,
    EmbeddingConfig,
)
from ..data.preprocessing import DataPreprocessor, DataConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
meta_learning_model: Optional[MetaLearningTrainer] = None
transformer_recommender: Optional[HybridTransformerRecommender] = None
data_preprocessor: Optional[DataPreprocessor] = None
thread_pool: Optional[ThreadPoolExecutor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    global meta_learning_model, transformer_recommender, data_preprocessor, thread_pool

    # Startup
    logger.info("Starting Adaptive Learning Recommendation Engine...")

    # Initialize thread pool for CPU-intensive tasks
    thread_pool = ThreadPoolExecutor(max_workers=4)

    # Load or initialize models
    try:
        await load_models()
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Initialize with default models for demo
        await initialize_demo_models()

    yield

    # Shutdown
    logger.info("Shutting down...")
    if thread_pool:
        thread_pool.shutdown(wait=True)


app = FastAPI(
    title="Adaptive Learning Recommendation Engine",
    description="Meta-learning powered recommendation system with transformer embeddings",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class UserInteraction(BaseModel):
    item_id: str
    rating: Optional[float] = Field(default=1.0, ge=0.0, le=5.0)
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RecommendationRequest(BaseModel):
    user_id: str
    user_interactions: List[UserInteraction]
    num_recommendations: int = Field(default=10, ge=1, le=100)
    recommendation_type: str = Field(
        default="hybrid", pattern="^(hybrid|content|collaborative|meta)$"
    )
    exclude_items: Optional[List[str]] = Field(default_factory=list)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ItemData(BaseModel):
    item_id: str
    title: str
    description: str
    category: Optional[str] = None
    features: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    request_id: str
    user_id: str
    recommendations: List[Dict[str, Any]]
    model_type: str
    processing_time_ms: float
    metadata: Dict[str, Any]


class TrainingRequest(BaseModel):
    training_data_path: str
    model_type: str = Field(default="maml", pattern="^(maml|prototypical)$")
    epochs: int = Field(default=50, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=1, le=256)
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)


class ModelStatus(BaseModel):
    model_loaded: bool
    model_type: Optional[str]
    last_training: Optional[datetime]
    performance_metrics: Optional[Dict[str, float]]


# Async helper functions
async def load_models() -> None:
    """Load pre-trained models."""
    global meta_learning_model, transformer_recommender, data_preprocessor

    # This would load actual trained models in production
    # For now, we'll initialize demo models
    await initialize_demo_models()


async def initialize_demo_models() -> None:
    """Initialize models for demonstration."""
    global meta_learning_model, transformer_recommender, data_preprocessor

    # Initialize data preprocessor
    data_config = DataConfig()
    data_preprocessor = DataPreprocessor(data_config)

    # Initialize embedding model
    embedding_config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        use_gpu=torch.cuda.is_available(),
    )
    embedding_model = TransformerEmbeddingModel(embedding_config)

    # Initialize hybrid recommender
    transformer_recommender = HybridTransformerRecommender(embedding_model)

    # Initialize meta-learning model
    meta_learning_model = MetaLearningTrainer(model_type="maml")

    logger.info("Demo models initialized")


async def run_in_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run CPU-intensive function in thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, func, *args, **kwargs)


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "description": "Adaptive Learning Recommendation Engine API",
        "version": "1.0.0",
        "lifespan": "running",
        "endpoints": {
            "recommendations": "/recommend",
            "batch_recommendations": "/recommend/batch",
            "user_adaptation": "/adapt",
            "model_training": "/train",
            "model_status": "/status",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "models_loaded": {
            "meta_learning": meta_learning_model is not None,
            "transformer": transformer_recommender is not None,
            "preprocessor": data_preprocessor is not None,
        },
    }


@app.get("/status", response_model=ModelStatus)
async def get_model_status():
    """Get current model status and performance metrics."""

    return ModelStatus(
        model_loaded=meta_learning_model is not None
        and transformer_recommender is not None,
        model_type="hybrid_meta_learning",
        last_training=None,  # Would track actual training time
        performance_metrics={
            "accuracy": 0.85,  # Demo metrics
            "precision": 0.82,
            "recall": 0.78,
            "f1_score": 0.80,
        },
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations for a user."""

    start_time = asyncio.get_event_loop().time()
    request_id = str(uuid.uuid4())

    try:
        # Convert interactions to the format expected by models
        interactions = []
        for interaction in request.user_interactions:
            interactions.append(
                {
                    "item_id": interaction.item_id,
                    "rating": interaction.rating,
                    "timestamp": interaction.timestamp,
                    "user_id": request.user_id,
                }
            )

        # Generate recommendations based on type
        if request.recommendation_type == "meta":
            recommendations = await generate_meta_recommendations(
                request.user_id, interactions, request.num_recommendations
            )
        elif request.recommendation_type == "content":
            recommendations = await generate_content_recommendations(
                request.user_id,
                interactions,
                request.num_recommendations,
                request.exclude_items or [],
            )
        elif request.recommendation_type == "collaborative":
            recommendations = await generate_collaborative_recommendations(
                request.user_id, interactions, request.num_recommendations
            )
        else:  # hybrid
            recommendations = await generate_hybrid_recommendations(
                request.user_id,
                interactions,
                request.num_recommendations,
                request.exclude_items,
            )

        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000

        return RecommendationResponse(
            request_id=request_id,
            user_id=request.user_id,
            recommendations=recommendations,
            model_type=request.recommendation_type,
            processing_time_ms=processing_time,
            metadata={
                "num_user_interactions": len(interactions),
                "excluded_items": len(request.exclude_items or []),
                "context": request.context,
            },
        )

    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate recommendations: {str(e)}"
        )


@app.post("/recommend/batch")
async def get_batch_recommendations(requests: List[RecommendationRequest]):
    """Get recommendations for multiple users in batch."""

    if len(requests) > 100:
        raise HTTPException(
            status_code=400, detail="Batch size cannot exceed 100 requests"
        )

    # Process requests concurrently
    tasks = [get_recommendations(request) for request in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Separate successful results from errors
    successful_results = []
    errors = []

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            errors.append(
                {
                    "request_index": i,
                    "user_id": requests[i].user_id,
                    "error": str(result),
                }
            )
        else:
            successful_results.append(result)

    return {
        "successful_recommendations": successful_results,
        "errors": errors,
        "total_requests": len(requests),
        "successful_count": len(successful_results),
        "error_count": len(errors),
    }


@app.post("/adapt")
async def adapt_to_user(user_id: str, interactions: List[UserInteraction]):
    """Adapt the meta-learning model to a specific user's preferences."""

    if not meta_learning_model:
        raise HTTPException(status_code=503, detail="Meta-learning model not available")

    try:
        # Convert interactions to model format
        user_data = {
            "users": [user_id] * len(interactions),
            "items": [interaction.item_id for interaction in interactions],
            "ratings": [interaction.rating or 1.0 for interaction in interactions],
        }

        # Perform adaptation in thread pool
        await run_in_thread(
            meta_learning_model.adapt_to_new_user, user_data
        )

        return {
            "message": f"Model adapted for user {user_id}",
            "user_id": user_id,
            "num_interactions": len(interactions),
            "adaptation_successful": True,
        }

    except Exception as e:
        logger.error(f"Error adapting model for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to adapt model: {str(e)}")


@app.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training in the background."""

    training_id = str(uuid.uuid4())

    # Add training task to background
    background_tasks.add_task(
        run_training_task,
        training_id,
        request.training_data_path,
        request.model_type,
        request.epochs,
        request.batch_size,
        request.learning_rate,
    )

    return {
        "message": "Training started",
        "training_id": training_id,
        "model_type": request.model_type,
        "estimated_duration_minutes": request.epochs * 2,  # Rough estimate
    }


@app.post("/items/update")
async def update_items(items: List[ItemData]):
    """Update item catalog for content-based recommendations."""

    if not transformer_recommender:
        raise HTTPException(
            status_code=503, detail="Transformer recommender not available"
        )

    try:
        # Convert items to format expected by recommender
        items_dict = {
            item.item_id: f"{item.title} {item.description}" for item in items
        }

        # Update item embeddings in thread pool
        await run_in_thread(
            transformer_recommender.content_recommender.fit_items, items_dict
        )

        return {
            "message": "Item catalog updated successfully",
            "num_items": len(items),
            "updated_at": datetime.now(),
        }

    except Exception as e:
        logger.error(f"Error updating items: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update items: {str(e)}")


# Helper functions for different recommendation types
async def generate_meta_recommendations(
    user_id: str, interactions: List[Dict[str, Any]], k: int
) -> List[Dict[str, Any]]:
    """Generate recommendations using meta-learning."""

    if not meta_learning_model:
        return []

    # This would use the actual meta-learning model
    # For demo, return mock recommendations
    return [
        {
            "item_id": f"meta_item_{i}",
            "score": 0.9 - i * 0.1,
            "explanation": f"Meta-learning recommendation {i+1}",
            "confidence": 0.85,
        }
        for i in range(k)
    ]


async def generate_content_recommendations(
    user_id: str, interactions: List[Dict[str, Any]], k: int, exclude_items: List[str]
) -> List[Dict[str, Any]]:
    """Generate content-based recommendations."""

    if not transformer_recommender:
        return []

    # This would use the actual content-based recommender
    # For demo, return mock recommendations
    return [
        {
            "item_id": f"content_item_{i}",
            "score": 0.8 - i * 0.05,
            "explanation": f"Content-based recommendation {i+1}",
            "similarity_score": 0.75,
        }
        for i in range(k)
        if f"content_item_{i}" not in exclude_items
    ]


async def generate_collaborative_recommendations(
    user_id: str, interactions: List[Dict[str, Any]], k: int
) -> List[Dict[str, Any]]:
    """Generate collaborative filtering recommendations."""

    # Mock collaborative filtering recommendations
    return [
        {
            "item_id": f"collab_item_{i}",
            "score": 0.7 - i * 0.03,
            "explanation": "Users with similar preferences also liked this",
            "similar_users": 15,
        }
        for i in range(k)
    ]


async def generate_hybrid_recommendations(
    user_id: str, interactions: List[Dict[str, Any]], k: int, exclude_items: List[str]
) -> List[Dict[str, Any]]:
    """Generate hybrid recommendations combining multiple approaches."""

    if not transformer_recommender:
        # Fallback to mock recommendations
        return [
            {
                "item_id": f"hybrid_item_{i}",
                "score": 0.85 - i * 0.04,
                "explanation": (
                    "Hybrid recommendation combining content and collaborative signals"
                ),
                "content_score": 0.8,
                "collaborative_score": 0.7,
                "meta_score": 0.9,
            }
            for i in range(k)
            if f"hybrid_item_{i}" not in exclude_items
        ]

    # Use actual hybrid recommender
    try:
        recommendations = await run_in_thread(
            transformer_recommender.recommend, user_id, interactions, k
        )
        return recommendations
    except Exception as e:
        logger.error(f"Error in hybrid recommendations: {e}")
        return []


async def run_training_task(
    training_id: str,
    data_path: str,
    model_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    """Background task for model training."""

    logger.info(f"Starting training task {training_id}")

    try:
        # This would implement actual training logic
        # For demo, simulate training time
        await asyncio.sleep(10)  # Simulate training

        logger.info(f"Training task {training_id} completed successfully")

    except Exception as e:
        logger.error(f"Training task {training_id} failed: {e}")


# WebSocket endpoint for real-time recommendations
@app.websocket("/ws/recommendations/{user_id}")
async def websocket_recommendations(websocket: Any, user_id: str) -> None:
    """WebSocket endpoint for real-time recommendation updates."""

    await websocket.accept()

    try:
        while True:
            # Wait for user interaction data
            data = await websocket.receive_json()

            # Process interaction and generate recommendations
            interactions = data.get("interactions", [])
            num_recs = data.get("num_recommendations", 5)

            recommendations = await generate_hybrid_recommendations(
                user_id, interactions, num_recs, []
            )

            # Send recommendations back
            await websocket.send_json(
                {
                    "type": "recommendations",
                    "user_id": user_id,
                    "recommendations": recommendations,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info"
    )
