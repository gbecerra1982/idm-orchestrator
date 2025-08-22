"""
Agentic Table Search Module
Implements advanced search strategies for complex tables using Azure AI Search 2025
Focus: Retrieval optimization for complex table structures
"""
import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from azure.search.documents import SearchClient
from azure.search.documents.models import (
    VectorizedQuery,
    QueryType,
    SearchMode
)
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential

logger = logging.getLogger(__name__)


class AgenticTableSearch:
    """
    Implements agentic search patterns for complex table retrieval.
    Decomposes complex queries into parallel sub-queries for optimal performance.
    """
    
    def __init__(self, search_client: Optional[SearchClient] = None):
        """
        Initialize the agentic search handler.
        
        Args:
            search_client: Optional Azure Search client, will create if not provided
        """
        self.search_client = search_client or self._create_search_client()
        self.logger = logging.getLogger(__name__)
        
        # Configuration for agentic search
        self.max_parallel_queries = int(os.getenv("AGENTIC_MAX_PARALLEL_QUERIES", "5"))
        self.enable_semantic_search = os.getenv("ENABLE_SEMANTIC_SEARCH", "true").lower() == "true"
        self.use_vector_search = os.getenv("USE_VECTOR_SEARCH", "true").lower() == "true"
    
    def _create_search_client(self) -> Optional[SearchClient]:
        """Create Azure Search client if credentials are available."""
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        key = os.getenv("AZURE_SEARCH_KEY")
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "tables-index")
        
        if not endpoint:
            logger.warning("Azure Search endpoint not configured")
            return None
        
        try:
            if key:
                credential = AzureKeyCredential(key)
            else:
                credential = DefaultAzureCredential()
            
            return SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=credential,
                api_version="2024-11-01-preview"  # Latest API version for advanced features
            )
        except Exception as e:
            logger.error(f"Failed to create search client: {e}")
            return None
    
    async def execute_agentic_search(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agentic search by decomposing query into optimized sub-queries.
        
        Args:
            query: User's search query
            context: Additional context (filters, preferences, etc.)
            
        Returns:
            Search results with enhanced ranking and metadata
        """
        if not self.search_client:
            logger.warning("Search client not available")
            return {"results": [], "error": "Search service not configured"}
        
        try:
            # 1. Analyze query intent
            query_analysis = await self.analyze_query_intent(query, context)
            
            # 2. Generate parallel sub-queries
            sub_queries = self.generate_sub_queries(query_analysis)
            
            # 3. Execute queries in parallel
            results = await self.execute_parallel_queries(sub_queries)
            
            # 4. Merge and rank results
            merged_results = self.merge_and_rerank_results(results, query_analysis)
            
            # 5. Enhance with metadata
            enhanced_results = await self.enhance_results_with_metadata(merged_results)
            
            return {
                "results": enhanced_results,
                "query_analysis": query_analysis,
                "total_count": len(enhanced_results),
                "execution_time": query_analysis.get("execution_time", 0)
            }
            
        except Exception as e:
            logger.error(f"Error in agentic search: {e}")
            return {"results": [], "error": str(e)}
    
    async def analyze_query_intent(self, query: str, context: Dict) -> Dict[str, Any]:
        """
        Analyze query to understand search intent and requirements.
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Query analysis with intent classification
        """
        start_time = datetime.now()
        
        analysis = {
            "original_query": query,
            "query_lower": query.lower(),
            "context": context,
            "intents": [],
            "entities": [],
            "search_types": []
        }
        
        # Detect table-specific intents
        if any(term in query.lower() for term in ["table", "column", "row", "cell", "header"]):
            analysis["intents"].append("table_structure")
            analysis["search_types"].append("structure")
        
        if any(term in query.lower() for term in ["merged", "combined", "span", "fusionada", "combinada"]):
            analysis["intents"].append("merged_cells")
            analysis["search_types"].append("merged")
        
        if any(term in query.lower() for term in ["hierarchical", "nested", "multi-level", "jerárquico", "anidado"]):
            analysis["intents"].append("hierarchical_headers")
            analysis["search_types"].append("hierarchy")
        
        if any(term in query.lower() for term in ["total", "sum", "aggregate", "calculation", "formula"]):
            analysis["intents"].append("calculations")
            analysis["search_types"].append("aggregation")
        
        if any(term in query.lower() for term in ["compare", "versus", "vs", "comparison", "comparar"]):
            analysis["intents"].append("comparison")
            analysis["search_types"].append("comparison")
        
        # Extract potential entities (simplified - could use NER)
        import re
        
        # Look for quoted terms
        quoted_terms = re.findall(r'"([^"]*)"', query)
        analysis["entities"].extend(quoted_terms)
        
        # Look for numbers
        numbers = re.findall(r'\b\d+\.?\d*\b', query)
        if numbers:
            analysis["entities"].extend(numbers)
            analysis["search_types"].append("numeric")
        
        # Look for currency
        if re.search(r'[$€£¥₹]|USD|EUR|GBP', query):
            analysis["intents"].append("financial")
            analysis["search_types"].append("currency")
        
        # Default to content search if no specific intent detected
        if not analysis["search_types"]:
            analysis["search_types"].append("content")
        
        analysis["execution_time"] = (datetime.now() - start_time).total_seconds()
        
        return analysis
    
    def generate_sub_queries(self, query_analysis: Dict) -> List[Dict[str, Any]]:
        """
        Generate optimized sub-queries based on query analysis.
        
        Args:
            query_analysis: Analysis of the original query
            
        Returns:
            List of sub-query specifications
        """
        sub_queries = []
        original_query = query_analysis["original_query"]
        
        # 1. Content search query
        if "content" in query_analysis["search_types"] or not query_analysis["search_types"]:
            sub_queries.append({
                "type": "content",
                "search_text": original_query,
                "search_fields": ["content", "cells/value", "semanticContent/summary"],
                "query_type": "semantic" if self.enable_semantic_search else "simple",
                "top": 20
            })
        
        # 2. Structure search for table features
        if "structure" in query_analysis["search_types"]:
            sub_queries.append({
                "type": "structure",
                "search_text": "*",
                "filter": self._build_structure_filter(query_analysis),
                "select": ["id", "structure", "tableMetadata"],
                "top": 10
            })
        
        # 3. Merged cells specific search
        if "merged" in query_analysis["search_types"]:
            sub_queries.append({
                "type": "merged_cells",
                "search_text": "*",
                "filter": "structure/mergedCells/any()",
                "select": ["id", "structure/mergedCells", "cells"],
                "top": 10
            })
        
        # 4. Hierarchical headers search
        if "hierarchy" in query_analysis["search_types"]:
            sub_queries.append({
                "type": "hierarchical",
                "search_text": "*",
                "filter": "structure/headers/hierarchicalMap/any()",
                "select": ["id", "structure/headers", "tableMetadata"],
                "top": 10
            })
        
        # 5. Numeric/calculation search
        if "numeric" in query_analysis["search_types"] or "aggregation" in query_analysis["search_types"]:
            for number in query_analysis.get("entities", []):
                if number.replace('.', '').isdigit():
                    sub_queries.append({
                        "type": "numeric",
                        "search_text": number,
                        "search_fields": ["cells/value"],
                        "query_type": "simple",
                        "top": 10
                    })
        
        # 6. Entity-specific searches
        for entity in query_analysis.get("entities", [])[:3]:  # Limit to 3 entities
            if not entity.replace('.', '').isdigit():  # Skip numbers (already handled)
                sub_queries.append({
                    "type": "entity",
                    "search_text": f'"{entity}"',  # Exact match
                    "search_fields": ["cells/value", "structure/headers/level1", "structure/headers/level2"],
                    "query_type": "simple",
                    "search_mode": "all",
                    "top": 10
                })
        
        # Limit total sub-queries
        return sub_queries[:self.max_parallel_queries]
    
    def _build_structure_filter(self, query_analysis: Dict) -> str:
        """Build OData filter for structure-based search."""
        filters = []
        
        # Add filters based on detected intents
        if "merged_cells" in query_analysis["intents"]:
            filters.append("complexFeatures/has_merged_cells eq true")
        
        if "hierarchical_headers" in query_analysis["intents"]:
            filters.append("complexFeatures/has_hierarchical_headers eq true")
        
        # Add size filters if numbers detected
        for entity in query_analysis.get("entities", []):
            if entity.isdigit():
                num = int(entity)
                # Assume it might be row or column count
                filters.append(f"(tableMetadata/dimensions/rows eq {num} or tableMetadata/dimensions/columns eq {num})")
        
        return " and ".join(filters) if filters else None
    
    async def execute_parallel_queries(self, sub_queries: List[Dict]) -> List[Dict]:
        """
        Execute multiple queries in parallel for performance.
        
        Args:
            sub_queries: List of query specifications
            
        Returns:
            List of query results
        """
        if not self.search_client:
            return []
        
        async def execute_single_query(query_spec: Dict) -> Dict:
            """Execute a single search query."""
            try:
                search_kwargs = {
                    "search_text": query_spec.get("search_text", "*"),
                    "top": query_spec.get("top", 10)
                }
                
                # Add optional parameters
                if "filter" in query_spec and query_spec["filter"]:
                    search_kwargs["filter"] = query_spec["filter"]
                
                if "select" in query_spec:
                    search_kwargs["select"] = query_spec["select"]
                
                if "search_fields" in query_spec:
                    search_kwargs["search_fields"] = query_spec["search_fields"]
                
                if "query_type" in query_spec:
                    search_kwargs["query_type"] = query_spec["query_type"]
                
                if "search_mode" in query_spec:
                    search_kwargs["search_mode"] = query_spec["search_mode"]
                
                # Execute search
                results = self.search_client.search(**search_kwargs)
                
                return {
                    "query_type": query_spec["type"],
                    "results": list(results),
                    "count": len(list(results))
                }
                
            except Exception as e:
                logger.error(f"Error executing sub-query: {e}")
                return {
                    "query_type": query_spec.get("type", "unknown"),
                    "results": [],
                    "error": str(e)
                }
        
        # Execute queries in parallel
        tasks = [execute_single_query(q) for q in sub_queries]
        results = await asyncio.gather(*tasks)
        
        return results
    
    def merge_and_rerank_results(self, results: List[Dict], query_analysis: Dict) -> List[Dict]:
        """
        Merge results from multiple queries and rerank by relevance.
        
        Args:
            results: Results from parallel queries
            query_analysis: Original query analysis
            
        Returns:
            Merged and ranked results
        """
        # Collect all unique results
        seen_ids = set()
        merged_results = []
        
        # Score boosts for different query types
        type_boosts = {
            "content": 1.0,
            "entity": 1.5,
            "numeric": 1.3,
            "structure": 0.8,
            "merged_cells": 1.2,
            "hierarchical": 1.2
        }
        
        for result_set in results:
            query_type = result_set.get("query_type", "unknown")
            boost = type_boosts.get(query_type, 1.0)
            
            for item in result_set.get("results", []):
                item_id = item.get("id") or item.get("table_id", "")
                
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    
                    # Calculate relevance score
                    base_score = item.get("@search.score", 1.0) if hasattr(item, "@search.score") else 1.0
                    item["relevance_score"] = base_score * boost
                    item["matched_query_types"] = [query_type]
                    
                    merged_results.append(item)
                elif item_id in seen_ids:
                    # Update existing result with additional query type match
                    for existing in merged_results:
                        if existing.get("id") == item_id or existing.get("table_id") == item_id:
                            existing["matched_query_types"].append(query_type)
                            existing["relevance_score"] += base_score * boost * 0.5  # Boost for multiple matches
                            break
        
        # Sort by relevance score
        merged_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Limit to top results
        max_results = int(os.getenv("MAX_SEARCH_RESULTS", "50"))
        return merged_results[:max_results]
    
    async def enhance_results_with_metadata(self, results: List[Dict]) -> List[Dict]:
        """
        Enhance search results with additional metadata for better interpretation.
        
        Args:
            results: Search results to enhance
            
        Returns:
            Enhanced results with metadata
        """
        enhanced = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Add interpretation hints based on matched query types
            hints = []
            if "merged_cells" in result.get("matched_query_types", []):
                hints.append("Contains merged cells - check cell spans for proper interpretation")
            
            if "hierarchical" in result.get("matched_query_types", []):
                hints.append("Has hierarchical headers - consider parent-child relationships")
            
            if "numeric" in result.get("matched_query_types", []):
                hints.append("Contains matching numeric values")
            
            enhanced_result["interpretation_hints"] = hints
            
            # Add complexity indicator
            complex_features = result.get("complexFeatures", {})
            complexity_score = sum([
                complex_features.get("has_merged_cells", False),
                complex_features.get("has_hierarchical_headers", False),
                complex_features.get("has_nested_structure", False),
                complex_features.get("has_multiple_header_rows", False)
            ])
            
            if complexity_score >= 3:
                enhanced_result["complexity"] = "high"
            elif complexity_score >= 1:
                enhanced_result["complexity"] = "medium"
            else:
                enhanced_result["complexity"] = "low"
            
            enhanced.append(enhanced_result)
        
        return enhanced


class TableQueryOptimizer:
    """
    Optimizes queries specifically for complex table structures.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def optimize_for_hierarchical_headers(self, query: str, header_levels: List[str]) -> str:
        """
        Optimize query for tables with hierarchical headers.
        
        Args:
            query: Original query
            header_levels: Known header hierarchy levels
            
        Returns:
            Optimized query
        """
        # Expand query to include header variations
        optimized = query
        
        for header in header_levels:
            if header.lower() in query.lower():
                # Add parent/child context
                optimized += f" OR (structure/headers/hierarchicalMap/any(h: h/parent eq '{header}' or h/children/any(c: c eq '{header}')))"
        
        return optimized
    
    def optimize_for_merged_cells(self, query: str) -> str:
        """
        Optimize query for tables with merged cells.
        
        Args:
            query: Original query
            
        Returns:
            Query optimized for merged cell search
        """
        # Add merged cell context to query
        return f"{query} AND structure/mergedCells/any(m: m/content ne null)"
    
    def generate_fallback_queries(self, query: str) -> List[str]:
        """
        Generate fallback queries for better coverage.
        
        Args:
            query: Original query
            
        Returns:
            List of fallback queries
        """
        fallbacks = [query]  # Original query first
        
        # Remove quotes for broader search
        if '"' in query:
            fallbacks.append(query.replace('"', ''))
        
        # Try with wildcards
        terms = query.split()
        if len(terms) <= 3:
            fallbacks.append(" ".join([f"{term}*" for term in terms]))
        
        # Try individual terms for very short queries
        if len(terms) == 2:
            fallbacks.extend(terms)
        
        return fallbacks[:5]  # Limit fallback queries