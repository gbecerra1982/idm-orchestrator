"""
Configuration Module for Complex Table Retrieval
Centralized configuration for all table processing components
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class RetrievalMode(Enum):
    """Retrieval modes for different scenarios."""
    STANDARD = "standard"
    ENHANCED = "enhanced"
    AGENTIC = "agentic"
    HYBRID = "hybrid"


@dataclass
class TableProcessingConfig:
    """Configuration for table processing features."""
    enable_complex_processing: bool = True
    enable_hierarchical_headers: bool = True
    enable_merged_cells: bool = True
    enable_borderless_detection: bool = True
    enable_document_intelligence: bool = False
    enable_agentic_search: bool = True
    enable_metrics: bool = True
    enable_mistral_ocr: bool = False
    max_tables_per_request: int = 10
    confidence_threshold: float = 0.7


@dataclass
class AzureSearchConfig:
    """Configuration for Azure AI Search."""
    endpoint: Optional[str] = None
    key: Optional[str] = None
    index_name: str = "tables-index"
    api_version: str = "2024-11-01-preview"
    semantic_config_name: str = "table-semantic-config"
    max_results: int = 50
    enable_semantic_search: bool = True
    enable_vector_search: bool = True
    enable_facets: bool = True


@dataclass
class DocumentIntelligenceConfig:
    """Configuration for Azure Document Intelligence."""
    endpoint: Optional[str] = None
    key: Optional[str] = None
    model: str = "prebuilt-layout"
    api_version: str = "2024-02-29-preview"
    enable_async: bool = True
    timeout_seconds: int = 60


@dataclass
class MistralOCRConfig:
    """Configuration for Mistral OCR integration with Azure AI Foundry."""
    # Azure AI Foundry OCR endpoint
    ocr_endpoint: Optional[str] = None
    ocr_api_key: Optional[str] = None
    ocr_model: str = "mistral-ocr-2503"
    
    # Optional: Azure AI Foundry small language model endpoint
    small_endpoint: Optional[str] = None
    small_api_key: Optional[str] = None
    small_model: str = "mistral-small-2503"
    
    # General settings
    enabled: bool = False
    max_concurrent_requests: int = 3
    timeout_seconds: int = 30
    use_for_complex_tables: bool = True
    confidence_threshold: float = 0.5


@dataclass
class PerformanceConfig:
    """Performance and optimization settings."""
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    max_parallel_queries: int = 5
    query_timeout_ms: int = 5000
    complex_table_timeout_multiplier: float = 1.5
    enable_query_optimization: bool = True
    enable_result_compression: bool = False


@dataclass
class MetricsConfig:
    """Metrics and monitoring configuration."""
    enable_logging: bool = True
    enable_performance_tracking: bool = True
    enable_quality_assessment: bool = True
    metrics_buffer_size: int = 1000
    export_metrics_interval_seconds: int = 300
    log_level: str = "INFO"


class ComplexTableConfig:
    """
    Main configuration class for complex table retrieval system.
    """
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        self.table_processing = self._load_table_processing_config()
        self.azure_search = self._load_azure_search_config()
        self.document_intelligence = self._load_document_intelligence_config()
        self.mistral_ocr = self._load_mistral_ocr_config()
        self.performance = self._load_performance_config()
        self.metrics = self._load_metrics_config()
        self.retrieval_mode = self._determine_retrieval_mode()
    
    def _load_table_processing_config(self) -> TableProcessingConfig:
        """Load table processing configuration."""
        return TableProcessingConfig(
            enable_complex_processing=self._get_bool_env("ENABLE_COMPLEX_TABLE_PROCESSING", True),
            enable_hierarchical_headers=self._get_bool_env("ENABLE_HIERARCHICAL_HEADERS", True),
            enable_merged_cells=self._get_bool_env("ENABLE_MERGED_CELLS", True),
            enable_borderless_detection=self._get_bool_env("ENABLE_BORDERLESS_DETECTION", True),
            enable_document_intelligence=self._get_bool_env("ENABLE_DOCUMENT_INTELLIGENCE", False),
            enable_agentic_search=self._get_bool_env("ENABLE_AGENTIC_TABLE_SEARCH", True),
            enable_metrics=self._get_bool_env("ENABLE_RETRIEVAL_METRICS", True),
            enable_mistral_ocr=self._get_bool_env("ENABLE_MISTRAL_OCR", False),
            max_tables_per_request=int(os.getenv("MAX_TABLES_PER_REQUEST", "10")),
            confidence_threshold=float(os.getenv("TABLE_CONFIDENCE_THRESHOLD", "0.7"))
        )
    
    def _load_azure_search_config(self) -> AzureSearchConfig:
        """Load Azure Search configuration."""
        return AzureSearchConfig(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            key=os.getenv("AZURE_SEARCH_KEY"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "tables-index"),
            api_version=os.getenv("AZURE_SEARCH_API_VERSION", "2024-11-01-preview"),
            semantic_config_name=os.getenv("AZURE_SEARCH_SEMANTIC_CONFIG", "table-semantic-config"),
            max_results=int(os.getenv("MAX_SEARCH_RESULTS", "50")),
            enable_semantic_search=self._get_bool_env("ENABLE_SEMANTIC_SEARCH", True),
            enable_vector_search=self._get_bool_env("ENABLE_VECTOR_SEARCH", True),
            enable_facets=self._get_bool_env("ENABLE_SEARCH_FACETS", True)
        )
    
    def _load_document_intelligence_config(self) -> DocumentIntelligenceConfig:
        """Load Document Intelligence configuration."""
        return DocumentIntelligenceConfig(
            endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
            key=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"),
            model=os.getenv("DOCUMENT_INTELLIGENCE_MODEL", "prebuilt-layout"),
            api_version=os.getenv("DOCUMENT_INTELLIGENCE_API_VERSION", "2024-02-29-preview"),
            enable_async=self._get_bool_env("DOCUMENT_INTELLIGENCE_ASYNC", True),
            timeout_seconds=int(os.getenv("DOCUMENT_INTELLIGENCE_TIMEOUT", "60"))
        )
    
    def _load_mistral_ocr_config(self) -> MistralOCRConfig:
        """Load Mistral OCR configuration for Azure AI Foundry."""
        return MistralOCRConfig(
            # Azure AI Foundry OCR endpoint
            ocr_endpoint=os.getenv("AZURE_MISTRAL_OCR_ENDPOINT"),
            ocr_api_key=os.getenv("AZURE_MISTRAL_OCR_API_KEY"),
            ocr_model=os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-2503"),
            
            # Optional: Azure AI Foundry small language model endpoint
            small_endpoint=os.getenv("AZURE_MISTRAL_SMALL_ENDPOINT"),
            small_api_key=os.getenv("AZURE_MISTRAL_SMALL_API_KEY"),
            small_model=os.getenv("MISTRAL_SMALL_MODEL", "mistral-small-2503"),
            
            # General settings
            enabled=self._get_bool_env("ENABLE_MISTRAL_OCR", False),
            max_concurrent_requests=int(os.getenv("MISTRAL_MAX_CONCURRENT", "3")),
            timeout_seconds=int(os.getenv("MISTRAL_TIMEOUT", "30")),
            use_for_complex_tables=self._get_bool_env("MISTRAL_USE_FOR_COMPLEX", True),
            confidence_threshold=float(os.getenv("MISTRAL_CONFIDENCE_THRESHOLD", "0.5"))
        )
    
    def _load_performance_config(self) -> PerformanceConfig:
        """Load performance configuration."""
        return PerformanceConfig(
            cache_enabled=self._get_bool_env("ENABLE_CACHE", True),
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
            max_parallel_queries=int(os.getenv("AGENTIC_MAX_PARALLEL_QUERIES", "5")),
            query_timeout_ms=int(os.getenv("QUERY_TIMEOUT_MS", "5000")),
            complex_table_timeout_multiplier=float(os.getenv("COMPLEX_TABLE_TIMEOUT_MULTIPLIER", "1.5")),
            enable_query_optimization=self._get_bool_env("ENABLE_QUERY_OPTIMIZATION", True),
            enable_result_compression=self._get_bool_env("ENABLE_RESULT_COMPRESSION", False)
        )
    
    def _load_metrics_config(self) -> MetricsConfig:
        """Load metrics configuration."""
        return MetricsConfig(
            enable_logging=self._get_bool_env("ENABLE_METRICS_LOGGING", True),
            enable_performance_tracking=self._get_bool_env("ENABLE_PERFORMANCE_TRACKING", True),
            enable_quality_assessment=self._get_bool_env("ENABLE_QUALITY_ASSESSMENT", True),
            metrics_buffer_size=int(os.getenv("METRICS_BUFFER_SIZE", "1000")),
            export_metrics_interval_seconds=int(os.getenv("METRICS_EXPORT_INTERVAL", "300")),
            log_level=os.getenv("LOGLEVEL", "INFO").upper()
        )
    
    def _determine_retrieval_mode(self) -> RetrievalMode:
        """Determine the retrieval mode based on configuration."""
        mode_str = os.getenv("RETRIEVAL_MODE", "").lower()
        
        if mode_str == "agentic":
            return RetrievalMode.AGENTIC
        elif mode_str == "enhanced":
            return RetrievalMode.ENHANCED
        elif mode_str == "standard":
            return RetrievalMode.STANDARD
        elif mode_str == "hybrid":
            return RetrievalMode.HYBRID
        else:
            # Auto-determine based on features
            if self.table_processing.enable_agentic_search and self.azure_search.endpoint:
                return RetrievalMode.AGENTIC
            elif self.table_processing.enable_complex_processing:
                return RetrievalMode.ENHANCED
            else:
                return RetrievalMode.STANDARD
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ["true", "1", "yes", "on"]
    
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags as a dictionary."""
        return {
            "complex_processing": self.table_processing.enable_complex_processing,
            "hierarchical_headers": self.table_processing.enable_hierarchical_headers,
            "merged_cells": self.table_processing.enable_merged_cells,
            "borderless_detection": self.table_processing.enable_borderless_detection,
            "document_intelligence": self.table_processing.enable_document_intelligence,
            "agentic_search": self.table_processing.enable_agentic_search,
            "semantic_search": self.azure_search.enable_semantic_search,
            "vector_search": self.azure_search.enable_vector_search,
            "metrics": self.metrics.enable_performance_tracking,
            "cache": self.performance.cache_enabled
        }
    
    def get_performance_thresholds(self) -> Dict[str, Any]:
        """Get performance thresholds for monitoring."""
        return {
            "excellent_time_ms": 500,
            "good_time_ms": 1000,
            "fair_time_ms": 2000,
            "poor_time_ms": self.performance.query_timeout_ms,
            "min_relevance_score": 0.7,
            "min_confidence": self.table_processing.confidence_threshold,
            "complex_multiplier": self.performance.complex_table_timeout_multiplier
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration and return status."""
        validation = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "mode": self.retrieval_mode.value
        }
        
        # Check Azure Search configuration
        if self.table_processing.enable_agentic_search:
            if not self.azure_search.endpoint:
                validation["warnings"].append("Agentic search enabled but Azure Search endpoint not configured")
                validation["is_valid"] = False
        
        # Check Document Intelligence configuration
        if self.table_processing.enable_document_intelligence:
            if not self.document_intelligence.endpoint:
                validation["warnings"].append("Document Intelligence enabled but endpoint not configured")
        
        # Check for conflicting settings
        if self.retrieval_mode == RetrievalMode.STANDARD and self.table_processing.enable_complex_processing:
            validation["warnings"].append("Standard mode selected but complex processing is enabled")
        
        # Performance warnings
        if self.performance.max_parallel_queries > 10:
            validation["warnings"].append(f"High parallel query limit ({self.performance.max_parallel_queries}) may cause performance issues")
        
        if self.performance.query_timeout_ms < 1000:
            validation["warnings"].append(f"Low query timeout ({self.performance.query_timeout_ms}ms) may cause failures for complex tables")
        
        return validation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "retrieval_mode": self.retrieval_mode.value,
            "table_processing": {
                "enable_complex_processing": self.table_processing.enable_complex_processing,
                "enable_hierarchical_headers": self.table_processing.enable_hierarchical_headers,
                "enable_merged_cells": self.table_processing.enable_merged_cells,
                "enable_borderless_detection": self.table_processing.enable_borderless_detection,
                "enable_document_intelligence": self.table_processing.enable_document_intelligence,
                "enable_agentic_search": self.table_processing.enable_agentic_search,
                "enable_metrics": self.table_processing.enable_metrics,
                "max_tables_per_request": self.table_processing.max_tables_per_request,
                "confidence_threshold": self.table_processing.confidence_threshold
            },
            "azure_search": {
                "endpoint_configured": bool(self.azure_search.endpoint),
                "index_name": self.azure_search.index_name,
                "api_version": self.azure_search.api_version,
                "max_results": self.azure_search.max_results,
                "enable_semantic_search": self.azure_search.enable_semantic_search,
                "enable_vector_search": self.azure_search.enable_vector_search
            },
            "document_intelligence": {
                "endpoint_configured": bool(self.document_intelligence.endpoint),
                "model": self.document_intelligence.model,
                "api_version": self.document_intelligence.api_version
            },
            "performance": {
                "cache_enabled": self.performance.cache_enabled,
                "max_parallel_queries": self.performance.max_parallel_queries,
                "query_timeout_ms": self.performance.query_timeout_ms
            },
            "metrics": {
                "enable_logging": self.metrics.enable_logging,
                "enable_performance_tracking": self.metrics.enable_performance_tracking,
                "log_level": self.metrics.log_level
            }
        }


# Global configuration instance
_config_instance = None


def get_config() -> ComplexTableConfig:
    """Get or create the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ComplexTableConfig()
    return _config_instance


def reload_config() -> ComplexTableConfig:
    """Reload configuration from environment."""
    global _config_instance
    _config_instance = ComplexTableConfig()
    return _config_instance