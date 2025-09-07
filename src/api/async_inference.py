"""Asynchronous inference engine using Asyncio + Joblib for
high-performance recommendations.
Handles concurrent requests and batch processing efficiently.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from joblib import Parallel, delayed
else:
    try:
        from joblib import Parallel, delayed
    except ImportError:
        Parallel = None
        delayed = None
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
import time
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for async inference engine."""

    max_concurrent_requests: int = 100
    batch_size: int = 32
    batch_timeout_ms: int = 50
    thread_pool_size: int = 4
    process_pool_size: int = 2
    enable_gpu_batching: bool = True
    cache_size: int = 1000
    enable_request_queuing: bool = True


class RequestBatch:
    """Batch of inference requests for efficient processing."""

    def __init__(self, max_size: int = 32, timeout_ms: int = 50):
        self.max_size = max_size
        self.timeout_ms = timeout_ms
        self.requests: List[Dict[str, Any]] = []
        self.futures: List[asyncio.Future[Any]] = []
        self.created_at = time.time()

    def add_request(self, request: Dict[str, Any], future: asyncio.Future[Any]) -> bool:
        """Add request to batch. Returns True if batch is full."""
        self.requests.append(request)
        self.futures.append(future)
        return len(self.requests) >= self.max_size

    def is_ready(self) -> bool:
        """Check if batch is ready for processing."""
        if len(self.requests) >= self.max_size:
            return True

        elapsed_ms = (time.time() - self.created_at) * 1000
        return elapsed_ms >= self.timeout_ms and len(self.requests) > 0

    def get_batch_data(
        self,
    ) -> Tuple[List[Dict[str, Any]], List[asyncio.Future[Any]]]:
        """Get batch data and futures."""
        return self.requests.copy(), self.futures.copy()


class InferenceCache:
    """LRU cache for inference results."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """Get cached result."""
        with self.lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None

    def put(self, key: str, value: Any) -> None:
        """Cache result."""
        with self.lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Remove least recently used
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]

            self.cache[key] = value
            self.access_order.append(key)

    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()


class AsyncInferenceEngine:
    """
    High-performance asynchronous inference engine for recommendation models.
    Supports batching, caching, and concurrent processing.
    """

    def __init__(self, config: InferenceConfig):
        self.config = config
        self.thread_pool = ThreadPoolExecutor(max_workers=config.thread_pool_size)
        self.process_pool = ProcessPoolExecutor(max_workers=config.process_pool_size)
        self.cache = InferenceCache(config.cache_size)

        # Request batching
        self.current_batch = None
        self.batch_lock = asyncio.Lock()
        self.batch_processor_task: Optional[asyncio.Task[None]] = None

        # Request queue for load balancing
        self.request_queue: asyncio.Queue[
            Tuple[Dict[str, Any], asyncio.Future[Any]]
        ] = asyncio.Queue(maxsize=config.max_concurrent_requests)
        self.batch_processors: List[asyncio.Task[None]] = []
        self.queue_processors: List[asyncio.Task[None]] = []

        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "batch_processed": 0,
            "avg_latency_ms": 0.0,
            "throughput_rps": 0.0,
        }
        self.metrics_lock = threading.Lock()

        # Model registry
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_lock = threading.Lock()

    async def start(self) -> None:
        """Start the inference engine."""
        logger.info("Starting async inference engine...")

        # Start batch processor
        self.batch_processor_task = asyncio.create_task(self._batch_processor())

        # Start queue processors if enabled
        if self.config.enable_request_queuing:
            for i in range(self.config.thread_pool_size):
                processor = asyncio.create_task(self._queue_processor(f"processor_{i}"))
                self.queue_processors.append(processor)

        logger.info("Async inference engine started")

    async def stop(self) -> None:
        """Stop the inference engine."""
        logger.info("Stopping async inference engine...")

        # Cancel batch processor
        if self.batch_processor_task:
            self.batch_processor_task.cancel()
            try:
                await self.batch_processor_task
            except asyncio.CancelledError:
                pass

        # Cancel queue processors
        for processor in self.queue_processors:
            processor.cancel()
            try:
                await processor
            except asyncio.CancelledError:
                pass

        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

        print("Async inference engine demo completed successfully!")

    def register_model(
        self,
        name: str,
        model: Any,
        inference_func: Callable[[Any, Dict[str, Any]], Any],
    ) -> None:
        """Register a model for inference."""
        with self.model_lock:
            self.models[name] = {
                "model": model,
                "inference_func": inference_func,
                "device": (
                    getattr(model, "device", "cpu")
                    if hasattr(model, "device")
                    else "cpu"
                ),
            }
        logger.info(f"Registered model: {name}")

    def unregister_model(self, name: str) -> None:
        """Unregister a model."""
        with self.model_lock:
            if name in self.models:
                del self.models[name]
                logger.info(f"Unregistered model: {name}")

    async def infer(
        self,
        model_name: str,
        request_data: Dict[str, Any],
        use_cache: bool = True,
        priority: int = 0,
    ) -> Any:
        """
        Perform asynchronous inference.

        Args:
            model_name: Name of the registered model
            request_data: Input data for inference
            use_cache: Whether to use caching
            priority: Request priority (higher = more important)

        Returns:
            Inference result
        """

        start_time = time.time()

        # Check cache first
        cache_key = None
        if use_cache:
            cache_key = self._generate_cache_key(model_name, request_data)
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self._update_metrics("cache_hit", time.time() - start_time)
                return cached_result

        # Create inference request
        request = {
            "model_name": model_name,
            "data": request_data,
            "priority": priority,
            "created_at": start_time,
            "cache_key": cache_key,
        }

        if self.config.enable_request_queuing:
            # Use request queue
            result = await self._queue_inference(request)
        else:
            # Direct inference
            result = await self._direct_inference(request)

        # Cache result
        if use_cache and cache_key:
            self.cache.put(cache_key, result)

        self._update_metrics("request_completed", time.time() - start_time)
        return result

    async def batch_infer(
        self,
        model_name: str,
        request_batch: List[Dict[str, Any]],
        use_cache: bool = True,
    ) -> List[Any]:
        """
        Perform batch inference for multiple requests.

        Args:
            model_name: Name of the registered model
            request_batch: List of request data
            use_cache: Whether to use caching

        Returns:
            List of inference results
        """

        # Check cache for each request
        results = []
        uncached_requests = []
        uncached_indices = []

        if use_cache:
            for i, request_data in enumerate(request_batch):
                cache_key = self._generate_cache_key(model_name, request_data)
                cached_result = self.cache.get(cache_key)

                if cached_result is not None:
                    results.append(cached_result)
                    self._update_metrics("cache_hit", 0)
                else:
                    results.append(None)  # Placeholder
                    uncached_requests.append(request_data)
                    uncached_indices.append(i)
        else:
            uncached_requests = request_batch
            uncached_indices = list(range(len(request_batch)))
            results = [None] * len(request_batch)

        # Process uncached requests in batch
        if uncached_requests:
            batch_results = await self._batch_inference(model_name, uncached_requests)

            # Fill in results and cache
            for idx, result in zip(uncached_indices, batch_results):
                results[idx] = result

                if use_cache:
                    cache_key = self._generate_cache_key(model_name, request_batch[idx])
                    self.cache.put(cache_key, result)

        return results

    async def _queue_inference(self, request: Dict[str, Any]) -> Any:
        """Process inference through request queue."""

        # Add to queue
        future: asyncio.Future[Any] = asyncio.Future()
        queue_item = (request, future)

        try:
            await self.request_queue.put(queue_item)
            return await future
        except asyncio.QueueFull:
            raise RuntimeError("Request queue is full")

    async def _direct_inference(self, request: Dict[str, Any]) -> Any:
        """Process inference directly."""

        model_name = request["model_name"]

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")

        model_info = self.models[model_name]

        # Run inference in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.thread_pool,
            model_info["inference_func"],
            model_info["model"],
            request["data"],
        )

        return result

    async def _batch_inference(
        self, model_name: str, request_batch: List[Dict[str, Any]]
    ) -> List[Any]:
        """Process batch inference."""

        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not registered")

        model_info = self.models[model_name]

        # Use joblib for parallel processing
        loop = asyncio.get_event_loop()

        def batch_inference_func() -> List[Any]:
            if Parallel is None or delayed is None:
                # Fallback to sequential processing if joblib not available
                return [
                    model_info["inference_func"](model_info["model"], data)
                    for data in request_batch
                ]
            return Parallel(n_jobs=self.config.thread_pool_size, backend="threading")(
                delayed(model_info["inference_func"])(model_info["model"], data)
                for data in request_batch
            )

        results = await loop.run_in_executor(self.process_pool, batch_inference_func)
        return results

    async def _batch_processor(self) -> None:
        """Background task for processing batched requests."""

        while True:
            try:
                async with self.batch_lock:
                    if self.current_batch and self.current_batch.is_ready():
                        # Process current batch
                        requests, futures = self.current_batch.get_batch_data()
                        self.current_batch = None

                        # Process batch outside the lock
                        asyncio.create_task(self._process_batch(requests, futures))

                # Small delay to prevent busy waiting
                await asyncio.sleep(0.001)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")

    async def _process_batch(
        self,
        requests: List[Dict[str, Any]],
        futures: List[asyncio.Future[Any]],
    ) -> None:
        """Process a batch of requests."""

        try:
            # Group requests by model
            model_groups = defaultdict(list)
            for i, request in enumerate(requests):
                model_name = request["model_name"]
                model_groups[model_name].append((i, request))

            # Process each model group
            results = [None] * len(requests)

            for model_name, model_requests in model_groups.items():
                indices = [item[0] for item in model_requests]
                request_data = [item[1]["data"] for item in model_requests]

                # Batch inference for this model
                model_results = await self._batch_inference(model_name, request_data)

                # Fill results
                for idx, result in zip(indices, model_results):
                    results[idx] = result

            # Set future results
            for future, result in zip(futures, results):
                if not future.cancelled():
                    future.set_result(result)

            self._update_metrics("batch_processed", 0)

        except Exception as e:
            # Set exception for all futures
            for future in futures:
                if not future.cancelled():
                    future.set_exception(e)
            logger.error(f"Error processing batch: {e}")

    async def _queue_processor(self, processor_id: str) -> None:
        """Background task for processing request queue."""

        logger.info(f"Started queue processor: {processor_id}")

        while True:
            try:
                # Get request from queue
                queue_item = await self.request_queue.get()
                request, future = queue_item

                if future.cancelled():
                    continue

                try:
                    # Process request
                    result = await self._direct_inference(request)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.request_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue processor {processor_id}: {e}")

    def _generate_cache_key(self, model_name: str, request_data: Dict[str, Any]) -> str:
        """Generate cache key for request."""

        # Simple hash-based key generation
        import hashlib
        import json

        key_data = {"model": model_name, "data": request_data}

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _update_metrics(self, metric_type: str, latency: float) -> None:
        """Update performance metrics."""

        with self.metrics_lock:
            if metric_type == "cache_hit":
                self.metrics["cache_hits"] += 1
            elif metric_type == "batch_processed":
                self.metrics["batch_processed"] += 1
            elif metric_type == "request_completed":
                self.metrics["total_requests"] += 1

                # Update average latency
                current_avg = self.metrics["avg_latency_ms"]
                total_requests = self.metrics["total_requests"]
                new_avg = (
                    (current_avg * (total_requests - 1)) + (latency * 1000)
                ) / total_requests
                self.metrics["avg_latency_ms"] = new_avg

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""

        with self.metrics_lock:
            metrics = self.metrics.copy()

            # Calculate cache hit rate
            total_requests = metrics["total_requests"]
            cache_hits = metrics["cache_hits"]
            metrics["cache_hit_rate"] = cache_hits / max(1, total_requests)

            return metrics

    def reset_metrics(self) -> None:
        """Reset performance metrics."""

        with self.metrics_lock:
            self.metrics = {
                "total_requests": 0,
                "cache_hits": 0,
                "batch_processed": 0,
                "avg_latency_ms": 0.0,
                "throughput_rps": 0.0,
            }


class ModelInferenceWrapper:
    """Wrapper for model inference functions to standardize interface."""

    def __init__(
        self,
        model: Any,
        preprocess_func: Optional[Callable[[Any], Any]] = None,
        postprocess_func: Optional[Callable[[Any], Any]] = None,
    ):
        self.model = model
        self.preprocess_func = preprocess_func
        self.postprocess_func = postprocess_func

    def __call__(self, model: Any, data: Any) -> Any:
        """Perform inference with optional pre/post processing."""

        # Preprocess
        if self.preprocess_func:
            data = self.preprocess_func(data)

        # Inference
        if hasattr(model, "predict"):
            result = model.predict(data)
        elif hasattr(model, "forward"):
            if isinstance(data, dict):
                result = model(**data)
            else:
                result = model(data)
        elif callable(model):
            result = model(data)
        else:
            raise ValueError("Model must have predict, forward method or be callable")

        # Postprocess
        if self.postprocess_func:
            result = self.postprocess_func(result)

        return result


# Example inference functions for different model types
def meta_learning_inference(model: Any, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Inference function for meta-learning models."""

    user_interactions = data.get("user_interactions", [])
    num_recommendations = data.get("num_recommendations", 10)

    # Convert to model format
    user_data = {
        "users": [data["user_id"]] * len(user_interactions),
        "items": [interaction["item_id"] for interaction in user_interactions],
        "ratings": [interaction["rating"] for interaction in user_interactions],
    }

    # Adapt model to user
    model.adapt_to_new_user(user_data)

    # Generate recommendations (mock implementation)
    recommendations = []
    for i in range(num_recommendations):
        recommendations.append(
            {
                "item_id": f"meta_rec_{i}",
                "score": 0.9 - i * 0.05,
                "explanation": f"Meta-learning recommendation {i+1}",
            }
        )

    return recommendations


def transformer_inference(model: Any, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Inference function for transformer-based models."""

    user_id = data["user_id"]
    user_interactions = data.get("user_interactions", [])
    num_recommendations = data.get("num_recommendations", 10)
    exclude_items = data.get("exclude_items", [])

    # Generate recommendations using hybrid approach
    recommendations = model.recommend(user_id, user_interactions, num_recommendations)

    # Filter excluded items
    filtered_recs = [
        rec for rec in recommendations if rec["item_id"] not in exclude_items
    ]

    return filtered_recs[:num_recommendations]


async def main() -> None:
    """Example usage of async inference engine."""

    # Configuration
    config = InferenceConfig(
        max_concurrent_requests=50,
        batch_size=16,
        batch_timeout_ms=100,
        thread_pool_size=4,
        enable_gpu_batching=True,
    )

    # Initialize engine
    engine = AsyncInferenceEngine(config)
    await engine.start()

    try:
        # Mock models for demonstration
        class MockModel:
            def predict(self, data: Any) -> Dict[str, str]:
                # Simulate inference time
                time.sleep(0.01)
                return {"prediction": f'result_for_{data.get("input", "unknown")}'}

        mock_model = MockModel()

        # Register model
        engine.register_model("mock_model", mock_model, lambda m, d: m.predict(d))

        # Test single inference
        result = await engine.infer("mock_model", {"input": "test_data"})
        print(f"Single inference result: {result}")

        # Test batch inference
        batch_requests = [{"input": f"batch_item_{i}"} for i in range(10)]
        batch_results = await engine.batch_infer("mock_model", batch_requests)
        print(f"Batch inference results: {len(batch_results)} items")

        # Test concurrent requests
        async def concurrent_request(i: int) -> Any:
            return await engine.infer("mock_model", {"input": f"concurrent_{i}"})

        concurrent_tasks = [concurrent_request(i) for i in range(20)]
        concurrent_results = await asyncio.gather(*concurrent_tasks)
        print(f"Concurrent inference results: {len(concurrent_results)} items")

        # Print metrics
        metrics = engine.get_metrics()
        print(f"Performance metrics: {metrics}")

    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
