"""
Mistral OCR Integration for Enhanced Table Retrieval via Azure AI Foundry
Leverages Mistral's OCR model (mistral-ocr-2503) for advanced table understanding during retrieval
Based on: https://github.com/azure-ai-foundry/foundry-samples/blob/main/samples/mistral/python/mistral-ocr-with-vlm.ipynb
Focus: Improving retrieval quality through better table comprehension
"""
import os
import json
import logging
import base64
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import aiohttp
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class MistralTableInsight:
    """Represents insights extracted by Mistral OCR from a table."""
    table_id: str
    semantic_summary: str
    key_relationships: List[Dict[str, Any]]
    extracted_values: Dict[str, Any]
    confidence_score: float
    structural_understanding: Dict[str, Any]


class MistralOCRTableAnalyzer:
    """
    Integrates Mistral OCR (Pixtral) for advanced table analysis during retrieval.
    Focuses on extracting semantic understanding that traditional OCR misses.
    """
    
    def __init__(self):
        """Initialize Mistral OCR analyzer for retrieval enhancement using Azure AI Foundry."""
        # Azure AI Foundry OCR endpoint configuration
        self.ocr_endpoint = os.getenv("AZURE_MISTRAL_OCR_ENDPOINT")
        self.ocr_api_key = os.getenv("AZURE_MISTRAL_OCR_API_KEY")
        self.ocr_model = os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-2503")
        
        # Optional: Small language model for additional processing
        self.small_endpoint = os.getenv("AZURE_MISTRAL_SMALL_ENDPOINT")
        self.small_api_key = os.getenv("AZURE_MISTRAL_SMALL_API_KEY")
        self.small_model = os.getenv("MISTRAL_SMALL_MODEL", "mistral-small-2503")
        
        self.enabled = os.getenv("ENABLE_MISTRAL_OCR", "false").lower() == "true"
        self.logger = logging.getLogger(__name__)
        
        if self.enabled and not (self.ocr_endpoint and self.ocr_api_key):
            self.logger.warning("Mistral OCR enabled but Azure AI Foundry endpoints not configured")
            self.enabled = False
    
    async def analyze_table_for_retrieval(self, 
                                         table_image: str,
                                         query: str,
                                         context: Dict[str, Any]) -> Optional[MistralTableInsight]:
        """
        Analyze table image using Mistral OCR to enhance retrieval quality.
        
        Args:
            table_image: Base64 encoded table image or URL
            query: User's search query
            context: Additional context (headers, metadata, etc.)
            
        Returns:
            MistralTableInsight with semantic understanding or None if not available
        """
        if not self.enabled:
            return None
        
        try:
            # Prepare specialized prompt for table analysis
            prompt = self._create_table_analysis_prompt(query, context)
            
            # Call Mistral API with image
            response = await self._call_mistral_api(table_image, prompt)
            
            if response:
                # Parse and structure the response
                insight = self._parse_mistral_response(response, query)
                return insight
            
        except Exception as e:
            self.logger.error(f"Error in Mistral OCR analysis: {e}")
            return None
    
    def _create_table_analysis_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """
        Create a specialized prompt for Mistral to analyze tables for retrieval.
        
        Args:
            query: User's search query
            context: Table context
            
        Returns:
            Optimized prompt for table analysis
        """
        # Extract context hints
        has_merged_cells = context.get("has_merged_cells", False)
        has_hierarchical_headers = context.get("has_hierarchical_headers", False)
        language = context.get("language", "English")
        
        prompt = f"""You are an expert table analyzer. Analyze this table image to help answer the following query:

Query: {query}

Your analysis should focus on:

1. **Semantic Understanding**: What is the table about? What story does it tell?
2. **Key Relationships**: Identify relationships between data points, especially those relevant to the query
3. **Hidden Patterns**: Find patterns that aren't immediately obvious from structure alone
4. **Contextual Values**: Extract specific values that might answer the query
5. **Structural Insights**: Understand the table's organization beyond basic rows/columns

Special considerations:
"""
        
        if has_merged_cells:
            prompt += "- This table has merged cells. Understand what values apply to multiple positions.\n"
        
        if has_hierarchical_headers:
            prompt += "- This table has hierarchical headers. Map the full hierarchy when extracting data.\n"
        
        if language != "English":
            prompt += f"- Respond in {language}.\n"
        
        prompt += """
Provide your analysis in the following JSON structure:
{
    "semantic_summary": "Brief description of what the table represents",
    "key_relationships": [
        {
            "type": "relationship type (calculation, comparison, hierarchy, etc.)",
            "description": "relationship description",
            "elements": ["element1", "element2"],
            "relevance_to_query": "how this relates to the user's query"
        }
    ],
    "extracted_values": {
        "direct_answer": "If the query can be directly answered",
        "related_values": {},
        "context_values": {}
    },
    "structural_insights": {
        "organization_pattern": "How the table is organized",
        "data_flow": "How data flows through the table",
        "implicit_groupings": "Groupings not explicitly marked"
    },
    "confidence": 0.0 to 1.0
}

Focus on information that would help answer the query accurately."""
        
        return prompt
    
    async def _call_mistral_api(self, image: str, prompt: str) -> Optional[Dict]:
        """
        Call Mistral API with image and prompt.
        
        Args:
            image: Base64 encoded image or URL
            prompt: Analysis prompt
            
        Returns:
            API response or None
        """
        headers = {
            "Authorization": f"Bearer {self.ocr_api_key}",
            "Content-Type": "application/json",
            "api-key": self.ocr_api_key  # Azure AI Foundry specific header
        }
        
        # Determine if image is URL or base64
        if image.startswith("http"):
            image_content = {"type": "image_url", "image_url": {"url": image}}
        else:
            # Ensure proper base64 format
            if not image.startswith("data:"):
                image = f"data:image/png;base64,{image}"
            image_content = {"type": "image_url", "image_url": {"url": image}}
        
        payload = {
            "model": self.ocr_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        image_content
                    ]
                }
            ],
            "temperature": 0.2,  # Low temperature for factual analysis
            "max_tokens": 2000
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.ocr_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        self.logger.error(f"Mistral API error: {response.status}")
                        return None
                        
        except asyncio.TimeoutError:
            self.logger.error("Mistral API timeout")
            return None
        except Exception as e:
            self.logger.error(f"Error calling Mistral API: {e}")
            return None
    
    def _parse_mistral_response(self, response: Dict, query: str) -> MistralTableInsight:
        """
        Parse Mistral API response into structured insight.
        
        Args:
            response: Mistral API response
            query: Original query for context
            
        Returns:
            Structured MistralTableInsight
        """
        try:
            # Extract the content from response
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # Parse JSON from content
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            parsed = json.loads(content)
            
            return MistralTableInsight(
                table_id=f"mistral_{hash(query)%10000}",
                semantic_summary=parsed.get("semantic_summary", ""),
                key_relationships=parsed.get("key_relationships", []),
                extracted_values=parsed.get("extracted_values", {}),
                confidence_score=float(parsed.get("confidence", 0.5)),
                structural_understanding=parsed.get("structural_insights", {})
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            self.logger.error(f"Error parsing Mistral response: {e}")
            
            # Return basic insight from raw content
            return MistralTableInsight(
                table_id=f"mistral_{hash(query)%10000}",
                semantic_summary=str(content)[:500],
                key_relationships=[],
                extracted_values={},
                confidence_score=0.3,
                structural_understanding={}
            )
    
    async def enhance_retrieval_with_semantic_understanding(self,
                                                           tables: List[Dict],
                                                           query: str) -> List[Dict]:
        """
        Enhance multiple tables with Mistral's semantic understanding for better retrieval.
        
        Args:
            tables: List of table data with images
            query: User's search query
            
        Returns:
            Enhanced tables with Mistral insights
        """
        if not self.enabled or not tables:
            return tables
        
        enhanced_tables = []
        
        # Process tables in parallel (with limit)
        max_concurrent = 3
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_table(table):
            async with semaphore:
                if "image" in table or "image_url" in table:
                    image = table.get("image") or table.get("image_url")
                    
                    insight = await self.analyze_table_for_retrieval(
                        image,
                        query,
                        table.get("metadata", {})
                    )
                    
                    if insight:
                        # Enhance table with Mistral insights
                        enhanced_table = table.copy()
                        enhanced_table["mistral_insights"] = {
                            "semantic_summary": insight.semantic_summary,
                            "key_relationships": insight.key_relationships,
                            "extracted_values": insight.extracted_values,
                            "confidence": insight.confidence_score,
                            "structural_understanding": insight.structural_understanding
                        }
                        
                        # Add direct answer if found
                        if insight.extracted_values.get("direct_answer"):
                            enhanced_table["potential_answer"] = insight.extracted_values["direct_answer"]
                        
                        return enhanced_table
                
                return table
        
        # Process all tables
        tasks = [process_table(table) for table in tables]
        enhanced_tables = await asyncio.gather(*tasks)
        
        # Sort by relevance if Mistral found answers
        enhanced_tables.sort(
            key=lambda t: t.get("mistral_insights", {}).get("confidence", 0),
            reverse=True
        )
        
        return enhanced_tables
    
    def generate_enhanced_prompt_context(self, insights: List[MistralTableInsight]) -> str:
        """
        Generate enhanced context for LLM prompt from Mistral insights.
        
        Args:
            insights: List of Mistral insights
            
        Returns:
            Enhanced context string for prompt
        """
        if not insights:
            return ""
        
        context_parts = ["[MISTRAL OCR INSIGHTS]"]
        
        for insight in insights:
            if insight.confidence_score > 0.5:  # Only include confident insights
                context_parts.append(f"\n[Table Analysis - Confidence: {insight.confidence_score:.2f}]")
                context_parts.append(f"Summary: {insight.semantic_summary}")
                
                if insight.extracted_values.get("direct_answer"):
                    context_parts.append(f"Potential Answer: {insight.extracted_values['direct_answer']}")
                
                if insight.key_relationships:
                    context_parts.append("Key Relationships:")
                    for rel in insight.key_relationships[:3]:  # Top 3 relationships
                        context_parts.append(f"  - {rel.get('description', '')}")
                        if rel.get("relevance_to_query"):
                            context_parts.append(f"    Relevance: {rel['relevance_to_query']}")
                
                if insight.structural_understanding:
                    org_pattern = insight.structural_understanding.get("organization_pattern")
                    if org_pattern:
                        context_parts.append(f"Organization: {org_pattern}")
        
        return "\n".join(context_parts)


class MistralTableRanker:
    """
    Uses Mistral to re-rank tables based on semantic relevance to query.
    """
    
    def __init__(self, analyzer: MistralOCRTableAnalyzer):
        """
        Initialize table ranker.
        
        Args:
            analyzer: MistralOCRTableAnalyzer instance
        """
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)
    
    async def rank_tables_by_relevance(self, 
                                      tables: List[Dict],
                                      query: str,
                                      top_k: int = 5) -> List[Dict]:
        """
        Rank tables by semantic relevance to query using Mistral.
        
        Args:
            tables: List of tables to rank
            query: User's search query
            top_k: Number of top tables to return
            
        Returns:
            Top-k ranked tables
        """
        if not self.analyzer.enabled or not tables:
            return tables[:top_k]
        
        # Create a ranking prompt
        ranking_prompt = f"""Given the user query: "{query}"
        
        Analyze these table summaries and rank them by relevance.
        Return a JSON array of table IDs ordered by relevance.
        
        Tables:
        """
        
        table_summaries = []
        for i, table in enumerate(tables[:10]):  # Limit to 10 for API constraints
            summary = table.get("summary", "No summary")
            headers = table.get("headers", [])
            table_summaries.append(f"Table {i}: {summary[:100]}... Headers: {headers[:5]}")
            ranking_prompt += f"\n{i}. {table_summaries[-1]}"
        
        ranking_prompt += "\n\nReturn JSON: {\"ranked_ids\": [most_relevant_id, ...]}"
        
        try:
            # Call Mistral for ranking
            response = await self.analyzer._call_mistral_api("", ranking_prompt)
            
            if response:
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                
                # Parse ranking
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                
                ranking = json.loads(content)
                ranked_ids = ranking.get("ranked_ids", list(range(len(tables))))
                
                # Reorder tables based on ranking
                ranked_tables = []
                for idx in ranked_ids[:top_k]:
                    if 0 <= idx < len(tables):
                        ranked_tables.append(tables[idx])
                
                # Add remaining tables if needed
                for table in tables:
                    if table not in ranked_tables and len(ranked_tables) < top_k:
                        ranked_tables.append(table)
                
                return ranked_tables
                
        except Exception as e:
            self.logger.error(f"Error in Mistral ranking: {e}")
            return tables[:top_k]


class MistralQueryEnhancer:
    """
    Uses Mistral to enhance queries for better table retrieval.
    """
    
    def __init__(self, analyzer: MistralOCRTableAnalyzer):
        """
        Initialize query enhancer.
        
        Args:
            analyzer: MistralOCRTableAnalyzer instance
        """
        self.analyzer = analyzer
        self.logger = logging.getLogger(__name__)
    
    async def enhance_table_query(self, 
                                 query: str,
                                 table_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance query with Mistral's understanding of table structures.
        
        Args:
            query: Original user query
            table_context: Context about available tables
            
        Returns:
            Enhanced query with additional search terms and structure hints
        """
        if not self.analyzer.enabled:
            return {"original": query, "enhanced": query}
        
        prompt = f"""Given a user query about tables: "{query}"
        
        And knowing these table characteristics:
        - Has merged cells: {table_context.get('has_merged_cells', False)}
        - Has hierarchical headers: {table_context.get('has_hierarchical_headers', False)}
        - Table domain: {table_context.get('domain', 'general')}
        
        Generate:
        1. Alternative phrasings that might find the same information
        2. Related terms and synonyms
        3. Structural hints (what table structure would contain this answer)
        
        Return JSON:
        {{
            "alternative_queries": [],
            "additional_terms": [],
            "structural_hints": {{
                "likely_headers": [],
                "likely_row_labels": [],
                "likely_in_merged_cells": true/false
            }}
        }}"""
        
        try:
            response = await self.analyzer._call_mistral_api("", prompt)
            
            if response:
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                
                enhancements = json.loads(content)
                
                return {
                    "original": query,
                    "enhanced": query,
                    "alternatives": enhancements.get("alternative_queries", []),
                    "additional_terms": enhancements.get("additional_terms", []),
                    "structural_hints": enhancements.get("structural_hints", {})
                }
                
        except Exception as e:
            self.logger.error(f"Error enhancing query with Mistral: {e}")
        
        return {"original": query, "enhanced": query}


class MistralRetrievalIntegration:
    """
    Main integration class that combines all Mistral capabilities for retrieval enhancement.
    """
    
    def __init__(self):
        """Initialize Mistral retrieval integration."""
        self.analyzer = MistralOCRTableAnalyzer()
        self.ranker = MistralTableRanker(self.analyzer)
        self.query_enhancer = MistralQueryEnhancer(self.analyzer)
        self.enabled = self.analyzer.enabled
        self.logger = logging.getLogger(__name__)
    
    async def enhance_table_retrieval_pipeline(self,
                                              query: str,
                                              retrieved_tables: List[Dict],
                                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete retrieval enhancement pipeline using Mistral OCR.
        
        Args:
            query: User's search query
            retrieved_tables: Tables retrieved from initial search
            context: Additional context
            
        Returns:
            Enhanced retrieval results with Mistral insights
        """
        if not self.enabled:
            return {
                "tables": retrieved_tables,
                "enhanced": False
            }
        
        try:
            # Step 1: Enhance query understanding
            enhanced_query = await self.query_enhancer.enhance_table_query(query, context)
            
            # Step 2: Analyze tables with Mistral OCR
            enhanced_tables = await self.analyzer.enhance_retrieval_with_semantic_understanding(
                retrieved_tables,
                query
            )
            
            # Step 3: Re-rank by relevance
            ranked_tables = await self.ranker.rank_tables_by_relevance(
                enhanced_tables,
                query,
                top_k=10
            )
            
            # Step 4: Generate enhanced context for LLM
            insights = [
                MistralTableInsight(**t["mistral_insights"])
                for t in ranked_tables
                if "mistral_insights" in t
            ]
            enhanced_context = self.analyzer.generate_enhanced_prompt_context(insights)
            
            return {
                "tables": ranked_tables,
                "enhanced": True,
                "query_enhancements": enhanced_query,
                "mistral_context": enhanced_context,
                "insights_count": len(insights),
                "confidence_scores": [i.confidence_score for i in insights]
            }
            
        except Exception as e:
            self.logger.error(f"Error in Mistral retrieval pipeline: {e}")
            return {
                "tables": retrieved_tables,
                "enhanced": False,
                "error": str(e)
            }
    
    def should_use_mistral(self, query: str, table_complexity: str) -> bool:
        """
        Determine if Mistral OCR should be used based on query and table complexity.
        
        Args:
            query: User query
            table_complexity: Complexity level of tables
            
        Returns:
            True if Mistral should be used
        """
        if not self.enabled:
            return False
        
        # Use Mistral for complex queries or tables
        complex_indicators = [
            "relationship", "pattern", "trend", "compare", "analysis",
            "correlation", "distribution", "breakdown", "composition"
        ]
        
        query_lower = query.lower()
        has_complex_query = any(indicator in query_lower for indicator in complex_indicators)
        
        return has_complex_query or table_complexity in ["high", "very_high"]