"""Configuration for the retrieval pipeline."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

class SearchMode(str, Enum):
    """Search mode for retrieval."""
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"  # Both dense and sparse

@dataclass
class ServiceConfig:
    """Configuration for external services."""
    dense_service_url: str = "http://localhost:4240"  # Port 4240 for dense service
    sparse_service_url: str = "http://localhost:4241"  # Port 4241 for sparse service
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        dense_url = os.getenv("DENSE_SERVICE_URL", "http://localhost:4240")
        sparse_url = os.getenv("SPARSE_SERVICE_URL", "http://localhost:4241")
        return cls(dense_service_url=dense_url, sparse_service_url=sparse_url)
    
@dataclass
class RerankerConfig:
    """Configuration for the reranker model."""
    # 优先使用本实验目录下已下载的本地模型（离线可用）；不存在时回退在线名称
    model_name: str = (
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bge-reranker-base")
        if os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "bge-reranker-base"))
        else "BAAI/bge-reranker-v2-m3"
    )
    device: str = None  # None = 自动检测（mps/cuda/cpu）；Windows 无 GPU 时落到 cpu
    batch_size: int = 32
    max_length: int = 8192  # Increased to match HARD_LIMIT in chunking
    use_fp16: bool = False  # CPU 上关闭半精度；GPU/MPS 环境可改回 True
    
@dataclass
class PipelineConfig:
    """Configuration for the retrieval pipeline."""
    services: ServiceConfig = field(default_factory=ServiceConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    
    # Retrieval settings
    default_top_k: int = 20  # Number of candidates to retrieve from each service
    rerank_top_k: int = 10  # Number of results after reranking

    # Fusion settings (see fusion.py)
    # "rrf": Reciprocal Rank Fusion (rank-only, robust); "weighted": weighted
    # min-max normalized score fusion; "avg_rank": legacy average-rank ordering.
    fusion_method: str = "rrf"
    rrf_k: int = 60  # RRF smoothing constant
    
    # Logging
    debug: bool = True
    show_scores: bool = True  # Show all scores in response for educational purposes
    
    # Server settings
    host: str = "0.0.0.0"
    port: int = 4242  # Default port for retrieval pipeline
    
    @classmethod
    def from_env(cls):
        """Create config from environment variables."""
        config = cls()
        if os.getenv("PIPELINE_PORT"):
            config.port = int(os.getenv("PIPELINE_PORT"))
        if os.getenv("PIPELINE_HOST"):
            config.host = os.getenv("PIPELINE_HOST")
        if os.getenv("DEBUG"):
            config.debug = os.getenv("DEBUG").lower() == "true"
        return config
