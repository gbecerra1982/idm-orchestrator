"""
Table Processor Module
Handles extraction and processing of complex table structures
"""
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import ast
import uuid

logger = logging.getLogger(__name__)


class TableStructureExtractor:
    """
    Extracts and processes complex table structures with hierarchical headers,
    merged cells, and borderless tables.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def extract_table_structure(self, table_html: str, image_urls: List[str]) -> Dict[str, Any]:
        """
        Extracts comprehensive structure from HTML tables and associated images.
        
        Args:
            table_html: HTML content of the table
            image_urls: URLs of table images
            
        Returns:
            Dictionary containing structured table data
        """
        table_id = str(uuid.uuid4())
        
        structure = {
            "table_id": table_id,
            "html_content": table_html,
            "image_urls": image_urls,
            "structure": self._extract_structure_from_html(table_html),
            "metadata": self._extract_metadata(table_html),
            "semantic_content": await self._generate_semantic_content(table_html)
        }
        
        # Identify complex features
        structure["complex_features"] = {
            "has_merged_cells": self._detect_merged_cells(table_html),
            "has_hierarchical_headers": self._detect_hierarchical_headers(table_html),
            "has_borderless_sections": self._detect_borderless_sections(table_html),
            "has_nested_tables": self._detect_nested_tables(table_html)
        }
        
        return structure
    
    def _extract_structure_from_html(self, html: str) -> Dict[str, Any]:
        """Extract table structure from HTML."""
        import re
        from html.parser import HTMLParser
        
        class TableParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.structure = {
                    "rows": 0,
                    "columns": 0,
                    "headers": [],
                    "cells": [],
                    "merged_cells": []
                }
                self.current_row = -1
                self.current_col = 0
                self.in_header = False
                self.in_cell = False
                self.current_cell = {}
                
            def handle_starttag(self, tag, attrs):
                attrs_dict = dict(attrs)
                
                if tag == 'tr':
                    self.current_row += 1
                    self.current_col = 0
                    self.structure["rows"] = self.current_row + 1
                    
                elif tag == 'th':
                    self.in_header = True
                    self.current_cell = {
                        "row": self.current_row,
                        "column": self.current_col,
                        "is_header": True,
                        "content": "",
                        "rowspan": int(attrs_dict.get('rowspan', 1)),
                        "colspan": int(attrs_dict.get('colspan', 1))
                    }
                    
                    # Track merged cells
                    if self.current_cell["rowspan"] > 1 or self.current_cell["colspan"] > 1:
                        self.structure["merged_cells"].append({
                            "start_row": self.current_row,
                            "start_col": self.current_col,
                            "row_span": self.current_cell["rowspan"],
                            "col_span": self.current_cell["colspan"]
                        })
                    
                elif tag == 'td':
                    self.in_cell = True
                    self.current_cell = {
                        "row": self.current_row,
                        "column": self.current_col,
                        "is_header": False,
                        "content": "",
                        "rowspan": int(attrs_dict.get('rowspan', 1)),
                        "colspan": int(attrs_dict.get('colspan', 1))
                    }
                    
                    # Track merged cells
                    if self.current_cell["rowspan"] > 1 or self.current_cell["colspan"] > 1:
                        self.structure["merged_cells"].append({
                            "start_row": self.current_row,
                            "start_col": self.current_col,
                            "row_span": self.current_cell["rowspan"],
                            "col_span": self.current_cell["colspan"]
                        })
                    
            def handle_data(self, data):
                if self.in_header or self.in_cell:
                    self.current_cell["content"] += data.strip()
                    
            def handle_endtag(self, tag):
                if tag == 'th':
                    self.in_header = False
                    self.structure["headers"].append(self.current_cell)
                    self.structure["cells"].append(self.current_cell)
                    self.current_col += self.current_cell["colspan"]
                    self.structure["columns"] = max(self.structure["columns"], self.current_col)
                    
                elif tag == 'td':
                    self.in_cell = False
                    self.structure["cells"].append(self.current_cell)
                    self.current_col += self.current_cell["colspan"]
                    self.structure["columns"] = max(self.structure["columns"], self.current_col)
        
        parser = TableParser()
        parser.feed(html)
        return parser.structure
    
    def _extract_metadata(self, html: str) -> Dict[str, Any]:
        """Extract metadata from table HTML."""
        metadata = {
            "has_caption": "<caption" in html.lower(),
            "has_thead": "<thead" in html.lower(),
            "has_tbody": "<tbody" in html.lower(),
            "has_tfoot": "<tfoot" in html.lower(),
            "estimated_complexity": self._estimate_complexity(html)
        }
        return metadata
    
    def _estimate_complexity(self, html: str) -> str:
        """Estimate table complexity based on structure."""
        complexity_score = 0
        
        # Check for complexity indicators
        if "rowspan" in html.lower() or "colspan" in html.lower():
            complexity_score += 2
        if html.count("<table") > 1:  # Nested tables
            complexity_score += 3
        if "<thead" in html.lower() and html.count("<tr") > 10:
            complexity_score += 1
        if not "<border" in html.lower() or 'border="0"' in html.lower():
            complexity_score += 1
            
        if complexity_score >= 4:
            return "high"
        elif complexity_score >= 2:
            return "medium"
        else:
            return "low"
    
    def _detect_merged_cells(self, html: str) -> bool:
        """Detect if table has merged cells."""
        return "rowspan" in html.lower() or "colspan" in html.lower()
    
    def _detect_hierarchical_headers(self, html: str) -> bool:
        """Detect if table has hierarchical headers."""
        # Check for multiple header rows or complex header structure
        header_rows = html.lower().count("<th")
        total_rows = html.lower().count("<tr")
        
        # If more than 20% of rows contain headers, likely hierarchical
        if total_rows > 0 and header_rows / total_rows > 0.2:
            return True
            
        # Check for colspan in headers (common in hierarchical structures)
        if "<th" in html.lower() and "colspan" in html.lower():
            th_sections = re.findall(r'<th[^>]*>', html.lower())
            for th in th_sections:
                if "colspan" in th:
                    return True
                    
        return False
    
    def _detect_borderless_sections(self, html: str) -> bool:
        """Detect if table has borderless sections."""
        # Check for border="0" or no border attribute
        if 'border="0"' in html.lower():
            return True
        if "<table" in html.lower() and "border" not in html.lower():
            return True
        return False
    
    def _detect_nested_tables(self, html: str) -> bool:
        """Detect if table contains nested tables."""
        return html.count("<table") > 1
    
    async def _generate_semantic_content(self, html: str) -> Dict[str, Any]:
        """Generate semantic understanding of table content."""
        # Extract text content for analysis
        text_content = re.sub(r'<[^>]+>', ' ', html)
        text_content = ' '.join(text_content.split())
        
        semantic = {
            "summary": self._generate_summary(text_content),
            "data_types": self._detect_data_types(text_content),
            "key_patterns": self._detect_patterns(text_content),
            "relationships": self._detect_relationships(html)
        }
        
        return semantic
    
    def _generate_summary(self, text: str) -> str:
        """Generate a brief summary of table content."""
        # Simple heuristic-based summary
        words = text.split()[:50]  # First 50 words
        summary = ' '.join(words)
        if len(words) == 50:
            summary += "..."
        return summary
    
    def _detect_data_types(self, text: str) -> List[str]:
        """Detect types of data in the table."""
        data_types = []
        
        # Check for numbers
        if re.search(r'\d+\.?\d*', text):
            data_types.append("numeric")
            
        # Check for currency
        if re.search(r'[$€£¥₹]|\b(USD|EUR|GBP)\b', text):
            data_types.append("currency")
            
        # Check for dates
        if re.search(r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b', text):
            data_types.append("date")
            
        # Check for percentages
        if '%' in text:
            data_types.append("percentage")
            
        # Check for text
        if re.search(r'[a-zA-Z]{3,}', text):
            data_types.append("text")
            
        return data_types
    
    def _detect_patterns(self, text: str) -> List[str]:
        """Detect common patterns in table data."""
        patterns = []
        
        # Check for totals/subtotals
        if re.search(r'\b(total|subtotal|sum|aggregate)\b', text, re.IGNORECASE):
            patterns.append("aggregation")
            
        # Check for comparisons
        if re.search(r'\b(vs|versus|compared|comparison)\b', text, re.IGNORECASE):
            patterns.append("comparison")
            
        # Check for time series
        if re.search(r'\b(Q[1-4]|quarter|month|year|annual|daily|weekly)\b', text, re.IGNORECASE):
            patterns.append("time_series")
            
        return patterns
    
    def _detect_relationships(self, html: str) -> List[Dict[str, Any]]:
        """Detect relationships between table elements."""
        relationships = []
        
        # Detect calculation relationships (e.g., Total = Sum of rows)
        if "total" in html.lower():
            relationships.append({
                "type": "aggregation",
                "description": "Contains total/sum calculations"
            })
            
        # Detect hierarchical relationships
        if self._detect_hierarchical_headers(html):
            relationships.append({
                "type": "hierarchy",
                "description": "Contains hierarchical header structure"
            })
            
        return relationships


class TableEnhancer:
    """
    Enhances table data with additional context and structure.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def enhance_with_context(self, table_structure: Dict[str, Any], 
                            surrounding_text: str) -> Dict[str, Any]:
        """
        Enhance table structure with surrounding context.
        
        Args:
            table_structure: Extracted table structure
            surrounding_text: Text surrounding the table
            
        Returns:
            Enhanced table structure
        """
        enhanced = table_structure.copy()
        
        # Add contextual information
        enhanced["context"] = {
            "surrounding_text": surrounding_text[:500],  # First 500 chars
            "mentioned_headers": self._find_mentioned_headers(
                surrounding_text, 
                table_structure.get("structure", {}).get("headers", [])
            ),
            "referenced_values": self._find_referenced_values(
                surrounding_text,
                table_structure.get("structure", {}).get("cells", [])
            )
        }
        
        # Add interpretation hints
        enhanced["interpretation_hints"] = self._generate_interpretation_hints(
            table_structure,
            surrounding_text
        )
        
        return enhanced
    
    def _find_mentioned_headers(self, text: str, headers: List[Dict]) -> List[str]:
        """Find headers mentioned in surrounding text."""
        mentioned = []
        text_lower = text.lower()
        
        for header in headers:
            header_text = header.get("content", "").lower()
            if header_text and header_text in text_lower:
                mentioned.append(header_text)
                
        return mentioned
    
    def _find_referenced_values(self, text: str, cells: List[Dict]) -> List[str]:
        """Find cell values referenced in surrounding text."""
        referenced = []
        
        for cell in cells[:100]:  # Check first 100 cells for performance
            cell_content = str(cell.get("content", ""))
            if cell_content and len(cell_content) > 3:  # Skip very short values
                if cell_content in text:
                    referenced.append(cell_content)
                    
        return referenced[:10]  # Return max 10 references
    
    def _generate_interpretation_hints(self, structure: Dict, context: str) -> List[str]:
        """Generate hints for interpreting the table."""
        hints = []
        
        # Check for financial data
        if any(term in context.lower() for term in ["revenue", "cost", "profit", "financial"]):
            hints.append("Table appears to contain financial data")
            
        # Check for comparison data
        if any(term in context.lower() for term in ["comparison", "versus", "vs", "compared"]):
            hints.append("Table appears to be comparing different items or periods")
            
        # Check for time series
        if any(term in context.lower() for term in ["quarter", "month", "year", "trend"]):
            hints.append("Table appears to contain time series data")
            
        # Check complexity
        complexity = structure.get("metadata", {}).get("estimated_complexity", "low")
        if complexity == "high":
            hints.append("Complex table structure - pay attention to merged cells and hierarchical headers")
            
        return hints
    
    def create_markdown_representation(self, table_structure: Dict[str, Any]) -> str:
        """
        Create a markdown representation of the table.
        
        Args:
            table_structure: Table structure dictionary
            
        Returns:
            Markdown formatted table
        """
        structure = table_structure.get("structure", {})
        cells = structure.get("cells", [])
        rows = structure.get("rows", 0)
        columns = structure.get("columns", 0)
        
        if not cells or rows == 0 or columns == 0:
            return "| Empty Table |"
        
        # Create grid
        grid = [['' for _ in range(columns)] for _ in range(rows)]
        
        # Fill grid with cell content
        for cell in cells:
            row = cell.get("row", 0)
            col = cell.get("column", 0)
            content = cell.get("content", "")
            rowspan = cell.get("rowspan", 1)
            colspan = cell.get("colspan", 1)
            
            # Handle merged cells
            for r in range(row, min(row + rowspan, rows)):
                for c in range(col, min(col + colspan, columns)):
                    if r == row and c == col:
                        grid[r][c] = content
                    else:
                        grid[r][c] = "^^"  # Merged cell indicator
        
        # Build markdown
        markdown_lines = []
        
        for row_idx, row in enumerate(grid):
            # Create row
            row_str = "| " + " | ".join(cell if cell != "^^" else "" for cell in row) + " |"
            markdown_lines.append(row_str)
            
            # Add separator after first row (header)
            if row_idx == 0:
                separator = "|" + "|".join([" --- " for _ in range(columns)]) + "|"
                markdown_lines.append(separator)
        
        return "\n".join(markdown_lines)
    
    def create_json_representation(self, table_structure: Dict[str, Any]) -> str:
        """
        Create a JSON representation optimized for LLM consumption.
        
        Args:
            table_structure: Table structure dictionary
            
        Returns:
            JSON string representation
        """
        # Create simplified structure for LLM
        simplified = {
            "table_id": table_structure.get("table_id"),
            "dimensions": {
                "rows": table_structure.get("structure", {}).get("rows", 0),
                "columns": table_structure.get("structure", {}).get("columns", 0)
            },
            "complexity": table_structure.get("metadata", {}).get("estimated_complexity", "unknown"),
            "features": table_structure.get("complex_features", {}),
            "headers": self._simplify_headers(table_structure.get("structure", {}).get("headers", [])),
            "data_sample": self._get_data_sample(table_structure.get("structure", {}).get("cells", [])),
            "semantic_summary": table_structure.get("semantic_content", {}).get("summary", ""),
            "interpretation_hints": table_structure.get("interpretation_hints", [])
        }
        
        return json.dumps(simplified, indent=2, ensure_ascii=False)
    
    def _simplify_headers(self, headers: List[Dict]) -> List[str]:
        """Simplify headers for JSON representation."""
        return [h.get("content", "") for h in headers if h.get("content")]
    
    def _get_data_sample(self, cells: List[Dict], max_cells: int = 20) -> List[Dict]:
        """Get a sample of cell data."""
        sample = []
        for cell in cells[:max_cells]:
            if not cell.get("is_header", False):
                sample.append({
                    "row": cell.get("row"),
                    "col": cell.get("column"),
                    "value": cell.get("content", "")
                })
        return sample