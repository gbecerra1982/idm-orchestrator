"""
Test Suite for Complex Table Retrieval
Tests for validating retrieval quality with complex table structures
"""
import unittest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, List, Any

# Import modules to test
from shared.table_processor import TableStructureExtractor, TableEnhancer
from shared.complex_table_handler import (
    ComplexTableInterpreter,
    HierarchicalHeaderProcessor,
    MergedCellHandler,
    BorderlessTableHandler,
    TableComplexity
)
from shared.agentic_table_search import AgenticTableSearch, TableQueryOptimizer
from shared.table_retrieval_metrics import TableRetrievalMonitor, RetrievalQuality


class TestComplexTableRetrieval(unittest.TestCase):
    """Test complex table retrieval functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.table_extractor = TableStructureExtractor()
        self.table_enhancer = TableEnhancer()
        self.complex_interpreter = ComplexTableInterpreter()
        
        # Sample complex table HTML
        self.complex_table_html = """
        <table>
            <tr>
                <th rowspan="2">Category</th>
                <th colspan="4">2024 Revenue</th>
            </tr>
            <tr>
                <th>Q1</th>
                <th>Q2</th>
                <th>Q3</th>
                <th>Q4</th>
            </tr>
            <tr>
                <td>Product A</td>
                <td>$100,000</td>
                <td>$150,000</td>
                <td>$200,000</td>
                <td>$250,000</td>
            </tr>
            <tr>
                <td>Product B</td>
                <td>$80,000</td>
                <td>$90,000</td>
                <td>$110,000</td>
                <td>$120,000</td>
            </tr>
            <tr>
                <td colspan="1">Total</td>
                <td>$180,000</td>
                <td>$240,000</td>
                <td>$310,000</td>
                <td>$370,000</td>
            </tr>
        </table>
        """
        
        # Sample table data structure
        self.sample_table_data = {
            "table_id": "test_table_001",
            "metadata": {
                "row_count": 5,
                "column_count": 5,
                "cell_count": 25
            },
            "structure": {
                "headers": {
                    "level1": ["Category", "2024 Revenue"],
                    "level2": ["Q1", "Q2", "Q3", "Q4"],
                    "hierarchicalMap": [
                        {
                            "parent": "2024 Revenue",
                            "children": ["Q1", "Q2", "Q3", "Q4"],
                            "span": 4
                        }
                    ]
                },
                "mergedCells": [
                    {
                        "row_index": 0,
                        "column_index": 0,
                        "row_span": 2,
                        "column_span": 1,
                        "content": "Category"
                    },
                    {
                        "row_index": 0,
                        "column_index": 1,
                        "row_span": 1,
                        "column_span": 4,
                        "content": "2024 Revenue"
                    }
                ]
            },
            "cells": [
                {"row": 2, "column": 0, "content": "Product A", "kind": "content"},
                {"row": 2, "column": 1, "content": "$100,000", "kind": "content"},
                {"row": 2, "column": 2, "content": "$150,000", "kind": "content"},
                {"row": 2, "column": 3, "content": "$200,000", "kind": "content"},
                {"row": 2, "column": 4, "content": "$250,000", "kind": "content"}
            ],
            "complex_features": {
                "has_merged_cells": True,
                "has_hierarchical_headers": True,
                "has_row_headers": True,
                "has_nested_structure": False
            }
        }
    
    async def test_table_structure_extraction(self):
        """Test extraction of table structure from HTML."""
        structure = await self.table_extractor.extract_table_structure(
            self.complex_table_html,
            []
        )
        
        self.assertIsNotNone(structure)
        self.assertIn("structure", structure)
        self.assertIn("complex_features", structure)
        self.assertTrue(structure["complex_features"]["has_merged_cells"])
        self.assertTrue(structure["complex_features"]["has_hierarchical_headers"])
    
    def test_hierarchical_header_processing(self):
        """Test processing of hierarchical headers."""
        processor = HierarchicalHeaderProcessor()
        enhanced = processor.process_hierarchical_headers(self.sample_table_data)
        
        self.assertIn("hierarchical_headers", enhanced)
        self.assertIn("tree", enhanced["hierarchical_headers"])
        self.assertIn("paths", enhanced["hierarchical_headers"])
        self.assertGreater(enhanced["hierarchical_headers"]["depth"], 1)
    
    def test_merged_cell_handling(self):
        """Test handling of merged cells."""
        handler = MergedCellHandler()
        enhanced = handler.process_merged_cells(self.sample_table_data)
        
        self.assertIn("merged_cell_processing", enhanced)
        self.assertIn("merged_index", enhanced["merged_cell_processing"])
        self.assertEqual(enhanced["merged_cell_processing"]["total_merged"], 2)
        self.assertEqual(enhanced["merged_cell_processing"]["max_row_span"], 2)
        self.assertEqual(enhanced["merged_cell_processing"]["max_col_span"], 4)
    
    def test_complexity_assessment(self):
        """Test table complexity assessment."""
        complexity = self.complex_interpreter._assess_complexity(self.sample_table_data)
        
        self.assertIn(complexity, [TableComplexity.HIGH, TableComplexity.VERY_HIGH])
    
    def test_query_interpretation(self):
        """Test query interpretation for complex tables."""
        query = "What is the Q2 revenue for Product A?"
        interpreted = self.complex_interpreter.interpret_table_for_retrieval(
            self.sample_table_data,
            query
        )
        
        self.assertIn("query_interpretation", interpreted)
        self.assertIn("complexity_assessment", interpreted)
        self.assertIsInstance(interpreted["query_interpretation"], dict)
    
    def test_markdown_representation(self):
        """Test markdown representation generation."""
        markdown = self.table_enhancer.create_markdown_representation(self.sample_table_data)
        
        self.assertIsNotNone(markdown)
        self.assertIn("|", markdown)  # Check for table formatting
        self.assertIn("Product A", markdown)
    
    def test_json_representation(self):
        """Test JSON representation for LLM consumption."""
        json_repr = self.table_enhancer.create_json_representation(self.sample_table_data)
        
        self.assertIsNotNone(json_repr)
        parsed = json.loads(json_repr)
        self.assertIn("table_id", parsed)
        self.assertIn("dimensions", parsed)
        self.assertIn("features", parsed)
    
    async def test_agentic_search_query_analysis(self):
        """Test agentic search query analysis."""
        search = AgenticTableSearch()
        query = "Show me the total revenue for Q3 across all products"
        
        analysis = await search.analyze_query_intent(query, {})
        
        self.assertIn("intents", analysis)
        self.assertIn("search_types", analysis)
        self.assertIn("entities", analysis)
    
    def test_query_optimization(self):
        """Test query optimization for complex tables."""
        optimizer = TableQueryOptimizer()
        
        # Test hierarchical header optimization
        query = "revenue Q1"
        header_levels = ["2024 Revenue", "Q1", "Q2", "Q3", "Q4"]
        optimized = optimizer.optimize_for_hierarchical_headers(query, header_levels)
        
        self.assertIn("Q1", optimized)
        
        # Test merged cell optimization
        query = "total sales"
        optimized = optimizer.optimize_for_merged_cells(query)
        
        self.assertIn("merged", optimized.lower())
    
    def test_borderless_table_detection(self):
        """Test detection of borderless tables."""
        handler = BorderlessTableHandler()
        
        borderless_data = {
            "metadata": {"border": "0"},
            "cells": [
                {"row": 0, "column": 0, "content": "Header1"},
                {"row": 0, "column": 2, "content": "Header2"},  # Gap in columns
                {"row": 1, "column": 0, "content": "Data1"},
                {"row": 1, "column": 2, "content": "Data2"}
            ]
        }
        
        enhanced = handler.process_borderless_table(borderless_data)
        
        if "borderless_processing" in enhanced:
            self.assertTrue(enhanced["borderless_processing"]["is_borderless"])
    
    async def test_retrieval_metrics(self):
        """Test retrieval metrics tracking."""
        monitor = TableRetrievalMonitor()
        
        metrics = await monitor.track_retrieval(
            query="test query",
            results=[{"id": "1", "relevance_score": 0.9}],
            execution_time=0.5,
            query_analysis={"intents": ["table_query"], "search_method": "agentic"},
            table_features={
                "has_hierarchical_headers": True,
                "has_merged_cells": True,
                "row_count": 10,
                "column_count": 5
            }
        )
        
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.quality_assessment, RetrievalQuality.EXCELLENT)
        self.assertTrue(metrics.has_hierarchical_headers)
        self.assertTrue(metrics.has_merged_cells)
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        monitor = TableRetrievalMonitor()
        
        # Add some test metrics
        asyncio.run(monitor.track_retrieval(
            query="test1",
            results=[{"id": "1"}],
            execution_time=0.3,
            query_analysis={"intents": [], "search_method": "standard"},
            table_features={"has_hierarchical_headers": False, "has_merged_cells": False}
        ))
        
        summary = monitor.get_performance_summary()
        
        self.assertIn("total_queries", summary)
        self.assertIn("average_time_ms", summary)
        self.assertIn("quality_distribution", summary)
        self.assertEqual(summary["total_queries"], 1)


class TestTableEnhancements(unittest.TestCase):
    """Test table enhancement features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.enhancer = TableEnhancer()
        self.sample_structure = {
            "structure": {
                "headers": [
                    {"content": "Revenue"},
                    {"content": "Q1"},
                    {"content": "Q2"}
                ],
                "cells": [
                    {"content": "100000"},
                    {"content": "150000"}
                ]
            }
        }
    
    def test_context_enhancement(self):
        """Test enhancing table with surrounding context."""
        surrounding_text = "The following table shows quarterly revenue for our products. Revenue figures are in USD."
        
        enhanced = self.enhancer.enhance_with_context(
            self.sample_structure,
            surrounding_text
        )
        
        self.assertIn("context", enhanced)
        self.assertIn("surrounding_text", enhanced["context"])
        self.assertIn("mentioned_headers", enhanced["context"])
        self.assertIn("interpretation_hints", enhanced)
    
    def test_interpretation_hints_generation(self):
        """Test generation of interpretation hints."""
        context = "This financial report shows revenue, costs, and profit margins for Q1-Q4 2024."
        
        enhanced = self.enhancer.enhance_with_context(
            self.sample_structure,
            context
        )
        
        hints = enhanced.get("interpretation_hints", [])
        self.assertTrue(any("financial" in hint.lower() for hint in hints))
        self.assertTrue(any("time series" in hint.lower() for hint in hints))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete retrieval pipeline."""
    
    @patch('shared.agentic_table_search.SearchClient')
    async def test_end_to_end_retrieval(self, mock_search_client):
        """Test end-to-end retrieval with complex table."""
        # Mock search results
        mock_search_client.return_value.search.return_value = [
            {
                "id": "table_001",
                "content": "Financial data table",
                "@search.score": 0.95,
                "complexFeatures": {
                    "has_merged_cells": True,
                    "has_hierarchical_headers": True
                }
            }
        ]
        
        # Initialize components
        search = AgenticTableSearch(mock_search_client.return_value)
        
        # Execute search
        results = await search.execute_agentic_search(
            "Q2 revenue for Product A",
            {"document_type": "financial_report"}
        )
        
        self.assertIn("results", results)
        self.assertGreater(len(results["results"]), 0)
        self.assertIn("query_analysis", results)


def run_async_test(coro):
    """Helper to run async tests."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == "__main__":
    unittest.main()