# Adaptive Learning Recommendation Engine

A cutting-edge personalized recommendation system powered by meta-learning algorithms, transformer embeddings, and high-performance async inference. This system combines Model-Agnostic Meta-Learning (MAML), Prototypical Networks, and Hugging Face Transformers to deliver adaptive recommendations that quickly learn user preferences with minimal data.

## 🚀 Key Features

### Meta-Learning Algorithms
- **MAML (Model-Agnostic Meta-Learning)**: Enables rapid adaptation to new users with few interactions
- **Prototypical Networks**: Creates user preference prototypes for few-shot recommendation scenarios
- **Fast Adaptation**: Learns new user preferences in just a few gradient steps

### Transformer-Based Embeddings
- **Semantic Understanding**: Uses Hugging Face Transformers for content comprehension
- **Multiple Model Support**: Compatible with BERT, RoBERTa, DistilBERT, and Sentence Transformers
- **Hybrid Approach**: Combines collaborative filtering with content-based recommendations

### High-Performance API
- **FastAPI Framework**: Modern, fast web framework with automatic API documentation
- **Asynchronous Inference**: Concurrent request processing with Asyncio + Joblib
- **Batch Processing**: Efficient batch inference for high-throughput scenarios
- **Caching Layer**: LRU cache for improved response times

### Production-Ready Infrastructure
- **Docker Support**: Multi-stage builds for development and production
- **Monitoring**: Integrated Prometheus metrics and Grafana dashboards
- **Scalability**: Horizontal scaling with load balancing support
- **Database Integration**: PostgreSQL for metadata, Redis for caching

## 📋 Tech Stack

- **ML Frameworks**: PyTorch, Hugging Face Transformers, Scikit-learn
- **Data Processing**: Pandas, NumPy, DuckDB
- **Web Framework**: FastAPI, Uvicorn
- **Async Processing**: Asyncio, Joblib
- **Embeddings**: Sentence Transformers, FAISS
- **Databases**: PostgreSQL, Redis
- **Monitoring**: Prometheus, Grafana
- **Deployment**: Docker, Docker Compose

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client Apps   │───▶│   FastAPI API    │───▶│  Meta-Learning  │
│  (Web/Mobile)   │    │   (Async Inf.)   │    │     Models      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │   Transformer    │    │  Data Pipeline  │
                       │   Embeddings     │    │ (Pandas/DuckDB) │
                       └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Redis Cache +    │    │   PostgreSQL    │
                       │ FAISS Index      │    │   Metadata      │
                       └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose
- CUDA-compatible GPU (optional, for faster inference)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd Adaptive-Learning-Recommendation-Engine
```

2. **Create and activate virtual environment**:
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running with Docker (Recommended)

1. **Start all services**:
```bash
docker-compose up -d
```

2. **Access the API**:
- API Documentation: http://localhost:8000/docs
- Grafana Dashboard: http://localhost:3000 (admin/admin123)
- Prometheus Metrics: http://localhost:9090

### Development Setup

1. **Start development services**:
```bash
docker-compose --profile dev up -d
```

2. **Run the API locally**:
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

3. **Start Jupyter for experimentation**:
```bash
docker-compose --profile dev up jupyter
```

## 📖 Usage Examples

### Basic Recommendation Request

```python
import requests

# User interaction history
user_interactions = [
    {"item_id": "movie_123", "rating": 4.5, "timestamp": "2024-01-01T10:00:00"},
    {"item_id": "movie_456", "rating": 3.8, "timestamp": "2024-01-02T15:30:00"}
]

# Request recommendations
response = requests.post("http://localhost:8000/recommend", json={
    "user_id": "user_789",
    "user_interactions": user_interactions,
    "num_recommendations": 10,
    "recommendation_type": "hybrid"
})

recommendations = response.json()["recommendations"]
for rec in recommendations:
    print(f"Item: {rec['item_id']}, Score: {rec['hybrid_score']:.3f}")
```

### Meta-Learning Adaptation

```python
# Adapt model to new user
response = requests.post("http://localhost:8000/adapt", json={
    "user_id": "new_user_123",
    "interactions": user_interactions
})

print(f"Adaptation successful: {response.json()['adaptation_successful']}")
```

### Batch Processing

```python
# Process multiple users
batch_requests = [
    {
        "user_id": f"user_{i}",
        "user_interactions": user_interactions,
        "num_recommendations": 5
    }
    for i in range(10)
]

response = requests.post("http://localhost:8000/recommend/batch", json=batch_requests)
results = response.json()["successful_recommendations"]
```

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
ENVIRONMENT=production
LOG_LEVEL=info
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
MODEL_CACHE_SIZE=1000
BATCH_SIZE=32
MAX_CONCURRENT_REQUESTS=100

# Database Configuration
POSTGRES_URL=postgresql://user:pass@localhost:5432/recengine
REDIS_URL=redis://localhost:6379/0

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
USE_GPU=true
```

### Model Parameters

```python
# Meta-learning configuration
meta_config = {
    "inner_lr": 0.01,
    "meta_lr": 0.001,
    "inner_steps": 5,
    "embedding_dim": 64
}

# Transformer configuration
transformer_config = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "max_length": 512,
    "batch_size": 32,
    "use_gpu": True
}
```

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

- **Latency**: Average response time per request
- **Throughput**: Requests per second
- **Cache Hit Rate**: Percentage of cached responses
- **Model Accuracy**: Recommendation quality metrics
- **Resource Usage**: CPU, memory, and GPU utilization

Access metrics at: http://localhost:9090 (Prometheus) or http://localhost:3000 (Grafana)

## 🧪 Testing

### Run Unit Tests
```bash
pytest tests/ -v
```

### Run Integration Tests
```bash
pytest tests/integration/ -v
```

### Performance Testing
```bash
# Load testing with locust
pip install locust
locust -f tests/load_test.py --host=http://localhost:8000
```

## 📁 Project Structure

```
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # Main API endpoints
│   │   └── async_inference.py # Async inference engine
│   ├── data/                  # Data processing
│   │   └── preprocessing.py   # Data pipeline
│   └── models/                # ML models
│       ├── meta_learning.py   # MAML & Prototypical Networks
│       └── transformers_embeddings.py  # Transformer models
├── tests/                     # Test suite
├── notebooks/                 # Jupyter notebooks
├── monitoring/                # Prometheus & Grafana configs
├── docker-compose.yml         # Docker services
├── Dockerfile                 # Container definition
└── requirements.txt           # Python dependencies
```

## 🚀 Deployment

### Production Deployment

1. **Build production image**:
```bash
docker build --target production -t rec-engine:latest .
```

2. **Deploy with Docker Compose**:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

3. **Scale services**:
```bash
docker-compose up -d --scale recommendation-api=3
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recommendation-engine
spec:
  replicas: 3
  selector:
    matchLabels:
      app: recommendation-engine
  template:
    metadata:
      labels:
        app: recommendation-engine
    spec:
      containers:
      - name: api
        image: rec-engine:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## 🔍 Monitoring & Observability

### Key Metrics Tracked
- Request latency (p50, p95, p99)
- Error rates and status codes
- Model inference time
- Cache hit/miss ratios
- Resource utilization
- User engagement metrics

### Alerts Configuration
- High error rate (>5%)
- Increased latency (>500ms p95)
- Memory usage (>80%)
- Cache miss rate (>50%)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Use type hints
- Run `black` and `flake8` before committing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- PyTorch team for the ML framework
- FastAPI for the excellent web framework
- The open-source ML community

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the API docs at `/docs` endpoint

---

**Built with ❤️ for adaptive, intelligent recommendations**
A personalized recommendation engine powered by adaptive and meta-learning techniques.
