"""
Table Retrieval Metrics Module
Monitors and tracks retrieval performance for complex tables
Focus: Measuring retrieval quality and performance metrics
"""
import os
import json
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class RetrievalQuality(Enum):
    """Quality levels for retrieval results."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class TableRetrievalMetrics:
    """Metrics for a single table retrieval operation."""
    query_id: str
    timestamp: datetime
    query: str
    query_complexity: str
    table_complexity: str
    retrieval_time_ms: float
    results_count: int
    relevance_score: float
    has_hierarchical_headers: bool
    has_merged_cells: bool
    search_method: str
    quality_assessment: RetrievalQuality
    interpretation_accuracy: Optional[float] = None
    user_feedback: Optional[str] = None


class TableRetrievalMonitor:
    """
    Monitors and analyzes table retrieval performance for continuous improvement.
    """
    
    def __init__(self, enable_logging: bool = True):
        """
        Initialize the retrieval monitor.
        
        Args:
            enable_logging: Whether to enable detailed logging
        """
        self.enable_logging = enable_logging
        self.logger = logging.getLogger(__name__)
        self.metrics_buffer = []
        self.performance_thresholds = self._load_performance_thresholds()
    
    def _load_performance_thresholds(self) -> Dict[str, Any]:
        """Load performance thresholds from configuration."""
        return {
            "excellent_time_ms": 500,
            "good_time_ms": 1000,
            "fair_time_ms": 2000,
            "min_relevance_score": 0.7,
            "complex_table_time_multiplier": 1.5
        }
    
    async def track_retrieval(self, 
                             query: str,
                             results: List[Dict],
                             execution_time: float,
                             query_analysis: Dict,
                             table_features: Dict) -> TableRetrievalMetrics:
        """
        Track a table retrieval operation.
        
        Args:
            query: Original search query
            results: Retrieved results
            execution_time: Time taken in seconds
            query_analysis: Query analysis details
            table_features: Table complexity features
            
        Returns:
            Metrics for this retrieval
        """
        query_id = self._generate_query_id(query)
        
        # Calculate metrics
        metrics = TableRetrievalMetrics(
            query_id=query_id,
            timestamp=datetime.now(),
            query=query[:200],  # Truncate for storage
            query_complexity=self._assess_query_complexity(query_analysis),
            table_complexity=self._assess_table_complexity(table_features),
            retrieval_time_ms=execution_time * 1000,
            results_count=len(results),
            relevance_score=self._calculate_relevance_score(results),
            has_hierarchical_headers=table_features.get("has_hierarchical_headers", False),
            has_merged_cells=table_features.get("has_merged_cells", False),
            search_method=query_analysis.get("search_method", "standard"),
            quality_assessment=self._assess_retrieval_quality(
                execution_time * 1000,
                len(results),
                self._calculate_relevance_score(results),
                table_features
            )
        )
        
        # Buffer metrics
        self.metrics_buffer.append(metrics)
        
        # Log if enabled
        if self.enable_logging:
            self._log_metrics(metrics)
        
        return metrics
    
    def _generate_query_id(self, query: str) -> str:
        """Generate unique ID for query."""
        import hashlib
        timestamp = str(time.time())
        return hashlib.md5(f"{query}{timestamp}".encode()).hexdigest()[:12]
    
    def _assess_query_complexity(self, query_analysis: Dict) -> str:
        """Assess complexity of the query."""
        intents = query_analysis.get("intents", [])
        entities = query_analysis.get("entities", [])
        search_types = query_analysis.get("search_types", [])
        
        complexity_score = len(intents) + len(entities) * 0.5 + len(search_types)
        
        if complexity_score >= 5:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _assess_table_complexity(self, table_features: Dict) -> str:
        """Assess complexity of table structure."""
        score = 0
        
        if table_features.get("has_merged_cells"):
            score += 2
        if table_features.get("has_hierarchical_headers"):
            score += 2
        if table_features.get("has_nested_structure"):
            score += 3
        if table_features.get("is_borderless"):
            score += 1
        
        rows = table_features.get("row_count", 0)
        cols = table_features.get("column_count", 0)
        
        if rows > 50 or cols > 20:
            score += 2
        elif rows > 20 or cols > 10:
            score += 1
        
        if score >= 5:
            return "very_high"
        elif score >= 3:
            return "high"
        elif score >= 1:
            return "medium"
        else:
            return "low"
    
    def _calculate_relevance_score(self, results: List[Dict]) -> float:
        """Calculate average relevance score of results."""
        if not results:
            return 0.0
        
        scores = []
        for result in results:
            # Use Azure Search score if available
            if "@search.score" in result:
                scores.append(result["@search.score"])
            # Use custom relevance score if available
            elif "relevance_score" in result:
                scores.append(result["relevance_score"])
            else:
                # Default score based on position
                scores.append(1.0 / (len(scores) + 1))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _assess_retrieval_quality(self, 
                                 time_ms: float,
                                 result_count: int,
                                 relevance_score: float,
                                 table_features: Dict) -> RetrievalQuality:
        """Assess overall quality of retrieval."""
        thresholds = self.performance_thresholds
        
        # Adjust time thresholds for complex tables
        if self._assess_table_complexity(table_features) in ["high", "very_high"]:
            time_factor = thresholds["complex_table_time_multiplier"]
        else:
            time_factor = 1.0
        
        # Time-based assessment
        if time_ms <= thresholds["excellent_time_ms"] * time_factor:
            time_quality = 4
        elif time_ms <= thresholds["good_time_ms"] * time_factor:
            time_quality = 3
        elif time_ms <= thresholds["fair_time_ms"] * time_factor:
            time_quality = 2
        else:
            time_quality = 1
        
        # Relevance-based assessment
        if relevance_score >= 0.9:
            relevance_quality = 4
        elif relevance_score >= 0.7:
            relevance_quality = 3
        elif relevance_score >= 0.5:
            relevance_quality = 2
        else:
            relevance_quality = 1
        
        # Result count assessment
        if 0 < result_count <= 10:
            count_quality = 4
        elif 10 < result_count <= 20:
            count_quality = 3
        elif result_count > 20:
            count_quality = 2
        else:
            count_quality = 1
        
        # Calculate overall quality
        overall = (time_quality + relevance_quality * 2 + count_quality) / 4
        
        if overall >= 3.5:
            return RetrievalQuality.EXCELLENT
        elif overall >= 2.5:
            return RetrievalQuality.GOOD
        elif overall >= 1.5:
            return RetrievalQuality.FAIR
        else:
            return RetrievalQuality.POOR
    
    def _log_metrics(self, metrics: TableRetrievalMetrics):
        """Log metrics for monitoring."""
        log_data = {
            "query_id": metrics.query_id,
            "time_ms": metrics.retrieval_time_ms,
            "results": metrics.results_count,
            "relevance": metrics.relevance_score,
            "quality": metrics.quality_assessment.value,
            "table_complexity": metrics.table_complexity,
            "has_merged": metrics.has_merged_cells,
            "has_hierarchy": metrics.has_hierarchical_headers
        }
        
        self.logger.info(f"Table retrieval metrics: {json.dumps(log_data)}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of retrieval performance."""
        if not self.metrics_buffer:
            return {"message": "No metrics available"}
        
        # Calculate summary statistics
        total_queries = len(self.metrics_buffer)
        avg_time = sum(m.retrieval_time_ms for m in self.metrics_buffer) / total_queries
        avg_relevance = sum(m.relevance_score for m in self.metrics_buffer) / total_queries
        
        # Quality distribution
        quality_dist = {}
        for quality in RetrievalQuality:
            count = sum(1 for m in self.metrics_buffer if m.quality_assessment == quality)
            quality_dist[quality.value] = count
        
        # Complex table performance
        complex_metrics = [m for m in self.metrics_buffer if m.table_complexity in ["high", "very_high"]]
        complex_avg_time = sum(m.retrieval_time_ms for m in complex_metrics) / len(complex_metrics) if complex_metrics else 0
        
        # Hierarchical header performance
        hierarchical_metrics = [m for m in self.metrics_buffer if m.has_hierarchical_headers]
        hierarchical_avg_time = sum(m.retrieval_time_ms for m in hierarchical_metrics) / len(hierarchical_metrics) if hierarchical_metrics else 0
        
        # Merged cell performance
        merged_metrics = [m for m in self.metrics_buffer if m.has_merged_cells]
        merged_avg_time = sum(m.retrieval_time_ms for m in merged_metrics) / len(merged_metrics) if merged_metrics else 0
        
        return {
            "total_queries": total_queries,
            "average_time_ms": avg_time,
            "average_relevance_score": avg_relevance,
            "quality_distribution": quality_dist,
            "complex_table_avg_time_ms": complex_avg_time,
            "hierarchical_header_avg_time_ms": hierarchical_avg_time,
            "merged_cell_avg_time_ms": merged_avg_time,
            "performance_by_complexity": self._get_performance_by_complexity(),
            "recommendations": self._generate_recommendations()
        }
    
    def _get_performance_by_complexity(self) -> Dict[str, Dict]:
        """Get performance breakdown by table complexity."""
        complexity_groups = {}
        
        for metrics in self.metrics_buffer:
            complexity = metrics.table_complexity
            if complexity not in complexity_groups:
                complexity_groups[complexity] = {
                    "count": 0,
                    "total_time": 0,
                    "total_relevance": 0
                }
            
            complexity_groups[complexity]["count"] += 1
            complexity_groups[complexity]["total_time"] += metrics.retrieval_time_ms
            complexity_groups[complexity]["total_relevance"] += metrics.relevance_score
        
        # Calculate averages
        for complexity, data in complexity_groups.items():
            if data["count"] > 0:
                data["avg_time_ms"] = data["total_time"] / data["count"]
                data["avg_relevance"] = data["total_relevance"] / data["count"]
        
        return complexity_groups
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []
        
        if not self.metrics_buffer:
            return recommendations
        
        # Check average time
        avg_time = sum(m.retrieval_time_ms for m in self.metrics_buffer) / len(self.metrics_buffer)
        if avg_time > self.performance_thresholds["fair_time_ms"]:
            recommendations.append("Consider implementing caching for frequently accessed tables")
        
        # Check complex table performance
        complex_metrics = [m for m in self.metrics_buffer if m.table_complexity in ["high", "very_high"]]
        if complex_metrics:
            complex_poor = sum(1 for m in complex_metrics if m.quality_assessment == RetrievalQuality.POOR)
            if complex_poor / len(complex_metrics) > 0.3:
                recommendations.append("Complex table retrieval needs optimization - consider pre-processing")
        
        # Check relevance scores
        avg_relevance = sum(m.relevance_score for m in self.metrics_buffer) / len(self.metrics_buffer)
        if avg_relevance < self.performance_thresholds["min_relevance_score"]:
            recommendations.append("Low relevance scores - consider improving search query analysis")
        
        # Check for specific feature issues
        hierarchical_poor = sum(1 for m in self.metrics_buffer 
                              if m.has_hierarchical_headers and m.quality_assessment == RetrievalQuality.POOR)
        if hierarchical_poor > 5:
            recommendations.append("Hierarchical header handling needs improvement")
        
        merged_poor = sum(1 for m in self.metrics_buffer 
                         if m.has_merged_cells and m.quality_assessment == RetrievalQuality.POOR)
        if merged_poor > 5:
            recommendations.append("Merged cell interpretation needs enhancement")
        
        return recommendations
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics for analysis."""
        if format == "json":
            metrics_data = [asdict(m) for m in self.metrics_buffer]
            # Convert datetime to string
            for m in metrics_data:
                m["timestamp"] = m["timestamp"].isoformat()
            return json.dumps(metrics_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def clear_metrics_buffer(self):
        """Clear the metrics buffer."""
        self.metrics_buffer = []
        self.logger.info("Metrics buffer cleared")


class RetrievalPerformanceAnalyzer:
    """
    Analyzes retrieval performance patterns for optimization insights.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_performance_trends(self, metrics: List[TableRetrievalMetrics]) -> Dict[str, Any]:
        """
        Analyze performance trends over time.
        
        Args:
            metrics: List of retrieval metrics
            
        Returns:
            Performance trend analysis
        """
        if not metrics:
            return {"error": "No metrics to analyze"}
        
        # Sort by timestamp
        sorted_metrics = sorted(metrics, key=lambda x: x.timestamp)
        
        # Calculate trends
        trends = {
            "time_trend": self._calculate_time_trend(sorted_metrics),
            "relevance_trend": self._calculate_relevance_trend(sorted_metrics),
            "quality_trend": self._calculate_quality_trend(sorted_metrics),
            "complexity_impact": self._analyze_complexity_impact(sorted_metrics),
            "feature_impact": self._analyze_feature_impact(sorted_metrics)
        }
        
        return trends
    
    def _calculate_time_trend(self, metrics: List[TableRetrievalMetrics]) -> Dict:
        """Calculate retrieval time trend."""
        if len(metrics) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate moving average
        window_size = min(10, len(metrics) // 3)
        if window_size < 2:
            window_size = 2
        
        moving_avg = []
        for i in range(window_size, len(metrics) + 1):
            window = metrics[i-window_size:i]
            avg = sum(m.retrieval_time_ms for m in window) / len(window)
            moving_avg.append(avg)
        
        # Determine trend
        if len(moving_avg) >= 2:
            if moving_avg[-1] < moving_avg[0] * 0.9:
                trend = "improving"
            elif moving_avg[-1] > moving_avg[0] * 1.1:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "current_avg": moving_avg[-1] if moving_avg else 0,
            "initial_avg": moving_avg[0] if moving_avg else 0
        }
    
    def _calculate_relevance_trend(self, metrics: List[TableRetrievalMetrics]) -> Dict:
        """Calculate relevance score trend."""
        if len(metrics) < 2:
            return {"trend": "insufficient_data"}
        
        # Calculate moving average
        window_size = min(10, len(metrics) // 3)
        if window_size < 2:
            window_size = 2
        
        moving_avg = []
        for i in range(window_size, len(metrics) + 1):
            window = metrics[i-window_size:i]
            avg = sum(m.relevance_score for m in window) / len(window)
            moving_avg.append(avg)
        
        # Determine trend
        if len(moving_avg) >= 2:
            if moving_avg[-1] > moving_avg[0] * 1.05:
                trend = "improving"
            elif moving_avg[-1] < moving_avg[0] * 0.95:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "current_avg": moving_avg[-1] if moving_avg else 0,
            "initial_avg": moving_avg[0] if moving_avg else 0
        }
    
    def _calculate_quality_trend(self, metrics: List[TableRetrievalMetrics]) -> Dict:
        """Calculate quality distribution trend."""
        # Split metrics into time periods
        period_size = max(1, len(metrics) // 3)
        periods = [
            metrics[:period_size],
            metrics[period_size:period_size*2],
            metrics[period_size*2:]
        ]
        
        quality_by_period = []
        for period in periods:
            if period:
                dist = {}
                for quality in RetrievalQuality:
                    count = sum(1 for m in period if m.quality_assessment == quality)
                    dist[quality.value] = count / len(period)
                quality_by_period.append(dist)
        
        # Determine trend
        if len(quality_by_period) >= 2:
            excellent_trend = quality_by_period[-1].get("excellent", 0) - quality_by_period[0].get("excellent", 0)
            poor_trend = quality_by_period[-1].get("poor", 0) - quality_by_period[0].get("poor", 0)
            
            if excellent_trend > 0.1 and poor_trend < -0.05:
                trend = "improving"
            elif excellent_trend < -0.1 and poor_trend > 0.05:
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "quality_distribution": quality_by_period[-1] if quality_by_period else {}
        }
    
    def _analyze_complexity_impact(self, metrics: List[TableRetrievalMetrics]) -> Dict:
        """Analyze impact of table complexity on performance."""
        complexity_impact = {}
        
        # Group by complexity
        by_complexity = {}
        for m in metrics:
            if m.table_complexity not in by_complexity:
                by_complexity[m.table_complexity] = []
            by_complexity[m.table_complexity].append(m)
        
        # Calculate impact
        for complexity, group in by_complexity.items():
            complexity_impact[complexity] = {
                "count": len(group),
                "avg_time_ms": sum(m.retrieval_time_ms for m in group) / len(group),
                "avg_relevance": sum(m.relevance_score for m in group) / len(group),
                "quality_distribution": self._get_quality_distribution(group)
            }
        
        return complexity_impact
    
    def _analyze_feature_impact(self, metrics: List[TableRetrievalMetrics]) -> Dict:
        """Analyze impact of specific table features on performance."""
        feature_impact = {
            "hierarchical_headers": self._analyze_single_feature_impact(
                metrics, lambda m: m.has_hierarchical_headers
            ),
            "merged_cells": self._analyze_single_feature_impact(
                metrics, lambda m: m.has_merged_cells
            )
        }
        
        return feature_impact
    
    def _analyze_single_feature_impact(self, metrics: List[TableRetrievalMetrics], 
                                      feature_check) -> Dict:
        """Analyze impact of a single feature."""
        with_feature = [m for m in metrics if feature_check(m)]
        without_feature = [m for m in metrics if not feature_check(m)]
        
        result = {}
        
        if with_feature:
            result["with_feature"] = {
                "count": len(with_feature),
                "avg_time_ms": sum(m.retrieval_time_ms for m in with_feature) / len(with_feature),
                "avg_relevance": sum(m.relevance_score for m in with_feature) / len(with_feature)
            }
        
        if without_feature:
            result["without_feature"] = {
                "count": len(without_feature),
                "avg_time_ms": sum(m.retrieval_time_ms for m in without_feature) / len(without_feature),
                "avg_relevance": sum(m.relevance_score for m in without_feature) / len(without_feature)
            }
        
        # Calculate impact
        if with_feature and without_feature:
            result["time_impact_ms"] = result["with_feature"]["avg_time_ms"] - result["without_feature"]["avg_time_ms"]
            result["relevance_impact"] = result["with_feature"]["avg_relevance"] - result["without_feature"]["avg_relevance"]
        
        return result
    
    def _get_quality_distribution(self, metrics: List[TableRetrievalMetrics]) -> Dict:
        """Get quality distribution for a set of metrics."""
        dist = {}
        for quality in RetrievalQuality:
            count = sum(1 for m in metrics if m.quality_assessment == quality)
            dist[quality.value] = count / len(metrics) if metrics else 0
        return dist