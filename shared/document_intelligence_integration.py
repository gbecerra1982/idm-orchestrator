"""
Document Intelligence Integration Module
Handles extraction of complex tables from documents using Azure Document Intelligence
"""
import os
import logging
import asyncio
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse, unquote
import base64

from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential

logger = logging.getLogger(__name__)


class DocumentIntelligenceTableExtractor:
    """
    Extracts and processes tables from documents using Azure Document Intelligence.
    Specializes in handling complex table structures including merged cells,
    hierarchical headers, and borderless tables.
    """
    
    def __init__(self):
        self.endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        self.key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        self.model = os.getenv("DOCUMENT_INTELLIGENCE_MODEL", "prebuilt-layout")
        
        if not self.endpoint:
            logger.warning("Document Intelligence endpoint not configured")
            self.client = None
        else:
            try:
                if self.key:
                    credential = AzureKeyCredential(self.key)
                else:
                    credential = DefaultAzureCredential()
                
                self.client = DocumentIntelligenceClient(
                    endpoint=self.endpoint,
                    credential=credential,
                    api_version="2024-02-29-preview"  # Latest API version
                )
            except Exception as e:
                logger.error(f"Failed to initialize Document Intelligence client: {e}")
                self.client = None
    
    async def extract_tables_from_document(self, document_source: str) -> List[Dict[str, Any]]:
        """
        Extract all tables from a document.
        
        Args:
            document_source: URL or base64 encoded document
            
        Returns:
            List of extracted table structures
        """
        if not self.client:
            logger.warning("Document Intelligence client not available, skipping extraction")
            return []
        
        try:
            # Determine if source is URL or base64
            if document_source.startswith("http"):
                result = await self._analyze_from_url(document_source)
            else:
                result = await self._analyze_from_bytes(document_source)
            
            # Process tables
            tables = []
            if result and hasattr(result, 'tables'):
                for idx, table in enumerate(result.tables):
                    processed_table = await self._process_table(table, idx, document_source)
                    tables.append(processed_table)
            
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables from document: {e}")
            return []
    
    async def _analyze_from_url(self, url: str) -> Optional[AnalyzeResult]:
        """Analyze document from URL."""
        try:
            poller = self.client.begin_analyze_document(
                self.model,
                analyze_request={"url_source": url}
            )
            return await asyncio.create_task(
                asyncio.to_thread(poller.result)
            )
        except Exception as e:
            logger.error(f"Error analyzing document from URL: {e}")
            return None
    
    async def _analyze_from_bytes(self, base64_content: str) -> Optional[AnalyzeResult]:
        """Analyze document from base64 content."""
        try:
            # Decode base64 to bytes
            document_bytes = base64.b64decode(base64_content)
            
            poller = self.client.begin_analyze_document(
                self.model,
                analyze_request={"bytes_source": document_bytes}
            )
            return await asyncio.create_task(
                asyncio.to_thread(poller.result)
            )
        except Exception as e:
            logger.error(f"Error analyzing document from bytes: {e}")
            return None
    
    async def _process_table(self, table: Any, index: int, source: str) -> Dict[str, Any]:
        """
        Process a single table from Document Intelligence results.
        
        Args:
            table: Table object from Document Intelligence
            index: Table index in document
            source: Original document source
            
        Returns:
            Processed table structure
        """
        processed = {
            "table_id": f"{source}_table_{index}",
            "index": index,
            "metadata": self._extract_table_metadata(table),
            "structure": self._extract_table_structure(table),
            "cells": self._extract_cells(table),
            "complex_features": self._identify_complex_features(table),
            "semantic_content": await self._generate_semantic_content(table)
        }
        
        # Add bounding box information if available
        if hasattr(table, 'bounding_regions') and table.bounding_regions:
            processed["location"] = {
                "page": table.bounding_regions[0].page_number,
                "polygon": table.bounding_regions[0].polygon
            }
        
        return processed
    
    def _extract_table_metadata(self, table: Any) -> Dict[str, Any]:
        """Extract metadata from table."""
        metadata = {
            "row_count": table.row_count if hasattr(table, 'row_count') else 0,
            "column_count": table.column_count if hasattr(table, 'column_count') else 0,
            "cell_count": len(table.cells) if hasattr(table, 'cells') else 0
        }
        
        # Add spans information
        if hasattr(table, 'spans') and table.spans:
            metadata["spans"] = [
                {"offset": span.offset, "length": span.length}
                for span in table.spans
            ]
        
        return metadata
    
    def _extract_table_structure(self, table: Any) -> Dict[str, Any]:
        """Extract detailed table structure."""
        structure = {
            "headers": self._identify_headers(table),
            "merged_cells": self._identify_merged_cells(table),
            "hierarchical_map": self._build_hierarchical_map(table)
        }
        
        return structure
    
    def _identify_headers(self, table: Any) -> Dict[str, List]:
        """Identify and categorize headers."""
        headers = {
            "row_headers": [],
            "column_headers": [],
            "hierarchical_headers": []
        }
        
        if not hasattr(table, 'cells'):
            return headers
        
        for cell in table.cells:
            if hasattr(cell, 'kind'):
                if cell.kind == "rowHeader":
                    headers["row_headers"].append({
                        "content": cell.content if hasattr(cell, 'content') else "",
                        "row": cell.row_index if hasattr(cell, 'row_index') else 0,
                        "column": cell.column_index if hasattr(cell, 'column_index') else 0,
                        "row_span": cell.row_span if hasattr(cell, 'row_span') else 1,
                        "column_span": cell.column_span if hasattr(cell, 'column_span') else 1
                    })
                elif cell.kind == "columnHeader":
                    headers["column_headers"].append({
                        "content": cell.content if hasattr(cell, 'content') else "",
                        "row": cell.row_index if hasattr(cell, 'row_index') else 0,
                        "column": cell.column_index if hasattr(cell, 'column_index') else 0,
                        "row_span": cell.row_span if hasattr(cell, 'row_span') else 1,
                        "column_span": cell.column_span if hasattr(cell, 'column_span') else 1
                    })
        
        # Identify hierarchical headers (headers spanning multiple columns/rows)
        for header in headers["column_headers"]:
            if header["column_span"] > 1 or header["row_span"] > 1:
                headers["hierarchical_headers"].append(header)
        
        return headers
    
    def _identify_merged_cells(self, table: Any) -> List[Dict[str, Any]]:
        """Identify merged cells in the table."""
        merged_cells = []
        
        if not hasattr(table, 'cells'):
            return merged_cells
        
        for cell in table.cells:
            row_span = cell.row_span if hasattr(cell, 'row_span') else 1
            column_span = cell.column_span if hasattr(cell, 'column_span') else 1
            
            if row_span > 1 or column_span > 1:
                merged_cells.append({
                    "row_index": cell.row_index if hasattr(cell, 'row_index') else 0,
                    "column_index": cell.column_index if hasattr(cell, 'column_index') else 0,
                    "row_span": row_span,
                    "column_span": column_span,
                    "content": cell.content if hasattr(cell, 'content') else ""
                })
        
        return merged_cells
    
    def _build_hierarchical_map(self, table: Any) -> Dict[str, Any]:
        """Build a hierarchical map of headers."""
        hierarchical_map = {
            "levels": [],
            "relationships": []
        }
        
        headers = self._identify_headers(table)
        column_headers = headers["column_headers"]
        
        if not column_headers:
            return hierarchical_map
        
        # Group headers by row to identify levels
        header_rows = {}
        for header in column_headers:
            row = header["row"]
            if row not in header_rows:
                header_rows[row] = []
            header_rows[row].append(header)
        
        # Build levels
        for row in sorted(header_rows.keys()):
            hierarchical_map["levels"].append({
                "level": len(hierarchical_map["levels"]),
                "row": row,
                "headers": header_rows[row]
            })
        
        # Build relationships (parent-child)
        for i, level in enumerate(hierarchical_map["levels"][:-1]):
            next_level = hierarchical_map["levels"][i + 1]
            
            for parent in level["headers"]:
                # Find children based on column span overlap
                children = []
                parent_start = parent["column"]
                parent_end = parent_start + parent["column_span"]
                
                for child in next_level["headers"]:
                    child_start = child["column"]
                    child_end = child_start + child["column_span"]
                    
                    # Check if child falls within parent's span
                    if child_start >= parent_start and child_end <= parent_end:
                        children.append(child["content"])
                
                if children:
                    hierarchical_map["relationships"].append({
                        "parent": parent["content"],
                        "children": children,
                        "parent_level": i,
                        "child_level": i + 1
                    })
        
        return hierarchical_map
    
    def _extract_cells(self, table: Any) -> List[Dict[str, Any]]:
        """Extract all cells with their properties."""
        cells = []
        
        if not hasattr(table, 'cells'):
            return cells
        
        for cell in table.cells:
            cell_data = {
                "row": cell.row_index if hasattr(cell, 'row_index') else 0,
                "column": cell.column_index if hasattr(cell, 'column_index') else 0,
                "content": cell.content if hasattr(cell, 'content') else "",
                "kind": cell.kind if hasattr(cell, 'kind') else "content",
                "row_span": cell.row_span if hasattr(cell, 'row_span') else 1,
                "column_span": cell.column_span if hasattr(cell, 'column_span') else 1
            }
            
            # Add bounding polygon if available
            if hasattr(cell, 'bounding_regions') and cell.bounding_regions:
                cell_data["bounding_box"] = {
                    "page": cell.bounding_regions[0].page_number,
                    "polygon": cell.bounding_regions[0].polygon
                }
            
            cells.append(cell_data)
        
        return cells
    
    def _identify_complex_features(self, table: Any) -> Dict[str, bool]:
        """Identify complex features in the table."""
        features = {
            "has_merged_cells": False,
            "has_hierarchical_headers": False,
            "has_row_headers": False,
            "has_nested_structure": False,
            "has_multiple_header_rows": False,
            "is_borderless": False
        }
        
        if not hasattr(table, 'cells'):
            return features
        
        # Check for merged cells
        for cell in table.cells:
            if (hasattr(cell, 'row_span') and cell.row_span > 1) or \
               (hasattr(cell, 'column_span') and cell.column_span > 1):
                features["has_merged_cells"] = True
                break
        
        # Check for different header types
        header_rows = set()
        for cell in table.cells:
            if hasattr(cell, 'kind'):
                if cell.kind == "rowHeader":
                    features["has_row_headers"] = True
                elif cell.kind == "columnHeader":
                    if hasattr(cell, 'row_index'):
                        header_rows.add(cell.row_index)
        
        # Multiple header rows indicate hierarchical structure
        if len(header_rows) > 1:
            features["has_multiple_header_rows"] = True
            features["has_hierarchical_headers"] = True
        
        # Check for nested structure (based on content analysis)
        # This is a simplified check - could be enhanced
        if features["has_merged_cells"] and features["has_multiple_header_rows"]:
            features["has_nested_structure"] = True
        
        return features
    
    async def _generate_semantic_content(self, table: Any) -> Dict[str, Any]:
        """Generate semantic understanding of the table."""
        semantic = {
            "summary": "",
            "data_types": [],
            "key_insights": [],
            "patterns": []
        }
        
        if not hasattr(table, 'cells'):
            return semantic
        
        # Analyze content to determine data types and patterns
        content_samples = []
        numeric_count = 0
        text_count = 0
        
        for cell in table.cells[:100]:  # Sample first 100 cells
            if hasattr(cell, 'content') and cell.content:
                content = str(cell.content).strip()
                content_samples.append(content)
                
                # Check if numeric
                try:
                    float(content.replace(',', '').replace('$', '').replace('%', ''))
                    numeric_count += 1
                except:
                    text_count += 1
        
        # Determine primary data types
        if numeric_count > text_count:
            semantic["data_types"].append("primarily_numeric")
        else:
            semantic["data_types"].append("primarily_text")
        
        # Check for specific patterns
        if any('$' in s or '€' in s or '£' in s for s in content_samples):
            semantic["data_types"].append("currency")
            semantic["patterns"].append("financial_data")
        
        if any('%' in s for s in content_samples):
            semantic["data_types"].append("percentage")
            semantic["patterns"].append("percentage_data")
        
        # Generate summary based on structure
        row_count = table.row_count if hasattr(table, 'row_count') else 0
        col_count = table.column_count if hasattr(table, 'column_count') else 0
        
        semantic["summary"] = f"Table with {row_count} rows and {col_count} columns"
        
        # Add insights based on complex features
        features = self._identify_complex_features(table)
        if features["has_hierarchical_headers"]:
            semantic["key_insights"].append("Contains hierarchical header structure")
        if features["has_merged_cells"]:
            semantic["key_insights"].append("Contains merged cells requiring special interpretation")
        if features["has_row_headers"]:
            semantic["key_insights"].append("Contains row headers for data organization")
        
        return semantic


class TableStructureValidator:
    """
    Validates and corrects table structures extracted from Document Intelligence.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_table(self, table: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and correct table structure.
        
        Args:
            table: Extracted table structure
            
        Returns:
            Validated and corrected table structure
        """
        validated = table.copy()
        
        # Validate metadata
        validated["metadata"] = self._validate_metadata(table.get("metadata", {}))
        
        # Validate cell positions
        validated["cells"] = self._validate_cells(
            table.get("cells", []),
            validated["metadata"]["row_count"],
            validated["metadata"]["column_count"]
        )
        
        # Validate merged cells
        validated["structure"]["merged_cells"] = self._validate_merged_cells(
            table.get("structure", {}).get("merged_cells", []),
            validated["metadata"]["row_count"],
            validated["metadata"]["column_count"]
        )
        
        # Add validation status
        validated["validation"] = {
            "is_valid": True,
            "corrections_made": [],
            "warnings": []
        }
        
        return validated
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and correct table metadata."""
        validated = metadata.copy()
        
        # Ensure required fields exist
        validated.setdefault("row_count", 0)
        validated.setdefault("column_count", 0)
        validated.setdefault("cell_count", 0)
        
        # Validate counts are positive
        for key in ["row_count", "column_count", "cell_count"]:
            if validated[key] < 0:
                self.logger.warning(f"Correcting negative {key}: {validated[key]} -> 0")
                validated[key] = 0
        
        return validated
    
    def _validate_cells(self, cells: List[Dict], max_row: int, max_col: int) -> List[Dict]:
        """Validate cell positions and content."""
        validated_cells = []
        
        for cell in cells:
            validated_cell = cell.copy()
            
            # Ensure cell is within table bounds
            if validated_cell.get("row", 0) >= max_row:
                self.logger.warning(f"Cell row {validated_cell['row']} exceeds table bounds")
                validated_cell["row"] = max_row - 1
            
            if validated_cell.get("column", 0) >= max_col:
                self.logger.warning(f"Cell column {validated_cell['column']} exceeds table bounds")
                validated_cell["column"] = max_col - 1
            
            # Ensure required fields
            validated_cell.setdefault("content", "")
            validated_cell.setdefault("row_span", 1)
            validated_cell.setdefault("column_span", 1)
            validated_cell.setdefault("kind", "content")
            
            validated_cells.append(validated_cell)
        
        return validated_cells
    
    def _validate_merged_cells(self, merged_cells: List[Dict], max_row: int, max_col: int) -> List[Dict]:
        """Validate merged cell definitions."""
        validated_merged = []
        
        for merged in merged_cells:
            validated = merged.copy()
            
            # Ensure merged cell doesn't exceed table bounds
            row_end = validated.get("row_index", 0) + validated.get("row_span", 1)
            col_end = validated.get("column_index", 0) + validated.get("column_span", 1)
            
            if row_end > max_row:
                validated["row_span"] = max_row - validated.get("row_index", 0)
                self.logger.warning(f"Adjusted row_span for merged cell at ({validated.get('row_index')}, {validated.get('column_index')})")
            
            if col_end > max_col:
                validated["column_span"] = max_col - validated.get("column_index", 0)
                self.logger.warning(f"Adjusted column_span for merged cell at ({validated.get('row_index')}, {validated.get('column_index')})")
            
            validated_merged.append(validated)
        
        return validated_merged