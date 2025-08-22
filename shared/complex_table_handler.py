"""
Complex Table Handler Module
Specialized handlers for complex table structures during retrieval
Focus: Optimizing retrieval quality for tables with hierarchical headers, merged cells, and borderless structures
"""
import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TableComplexity(Enum):
    """Enum for table complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class HeaderHierarchy:
    """Represents hierarchical header structure."""
    level: int
    parent: Optional[str]
    children: List[str]
    span: int
    content: str


class HierarchicalHeaderProcessor:
    """
    Processes and interprets hierarchical headers in complex tables for optimal retrieval.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_hierarchical_headers(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process hierarchical headers to improve retrieval accuracy.
        
        Args:
            table_data: Table data with headers structure
            
        Returns:
            Enhanced table data with processed hierarchical relationships
        """
        headers = table_data.get("structure", {}).get("headers", {})
        
        if not headers:
            return table_data
        
        # Build header tree
        header_tree = self._build_header_tree(headers)
        
        # Generate header paths for better searching
        header_paths = self._generate_header_paths(header_tree)
        
        # Create header context map
        header_context = self._create_header_context_map(header_tree, table_data.get("cells", []))
        
        # Enhance table data
        enhanced_data = table_data.copy()
        enhanced_data["hierarchical_headers"] = {
            "tree": header_tree,
            "paths": header_paths,
            "context_map": header_context,
            "depth": self._calculate_hierarchy_depth(header_tree)
        }
        
        return enhanced_data
    
    def _build_header_tree(self, headers: Dict) -> List[HeaderHierarchy]:
        """Build a tree structure from header data."""
        tree = []
        
        # Process each level
        for level_key in ["level1", "level2", "level3"]:
            if level_key in headers:
                level_num = int(level_key[-1])
                for header in headers[level_key]:
                    if isinstance(header, str):
                        tree.append(HeaderHierarchy(
                            level=level_num,
                            parent=None,
                            children=[],
                            span=1,
                            content=header
                        ))
        
        # Process hierarchical map if available
        if "hierarchicalMap" in headers:
            for mapping in headers["hierarchicalMap"]:
                parent_content = mapping.get("parent", "")
                children = mapping.get("children", [])
                
                # Find parent in tree
                for node in tree:
                    if node.content == parent_content:
                        node.children = children
                        node.span = mapping.get("span", len(children))
                        break
        
        return tree
    
    def _generate_header_paths(self, header_tree: List[HeaderHierarchy]) -> List[str]:
        """Generate searchable paths from header tree."""
        paths = []
        
        def generate_path(node: HeaderHierarchy, parent_path: str = "") -> None:
            current_path = f"{parent_path}/{node.content}" if parent_path else node.content
            paths.append(current_path)
            
            for child_content in node.children:
                # Find child node
                for child_node in header_tree:
                    if child_node.content == child_content:
                        generate_path(child_node, current_path)
                        break
        
        for root_node in header_tree:
            if root_node.level == 1:  # Start from top level
                generate_path(root_node)
        
        return paths
    
    def _create_header_context_map(self, header_tree: List[HeaderHierarchy], cells: List[Dict]) -> Dict:
        """Create a map of headers to their data context."""
        context_map = {}
        
        for header in header_tree:
            # Find cells associated with this header
            associated_cells = []
            for cell in cells:
                cell_headers = cell.get("headers", {})
                if header.content in cell_headers.get("column", []) or \
                   header.content in cell_headers.get("row", []):
                    associated_cells.append({
                        "row": cell.get("row"),
                        "column": cell.get("column"),
                        "value": cell.get("content", "")
                    })
            
            context_map[header.content] = {
                "level": header.level,
                "parent": header.parent,
                "children": header.children,
                "associated_cells": associated_cells[:10],  # Limit for performance
                "cell_count": len(associated_cells)
            }
        
        return context_map
    
    def _calculate_hierarchy_depth(self, header_tree: List[HeaderHierarchy]) -> int:
        """Calculate the maximum depth of the header hierarchy."""
        if not header_tree:
            return 0
        return max(h.level for h in header_tree)
    
    def interpret_hierarchical_query(self, query: str, header_context: Dict) -> Dict[str, Any]:
        """
        Interpret a query in the context of hierarchical headers.
        
        Args:
            query: User query
            header_context: Header context map
            
        Returns:
            Interpreted query with header mappings
        """
        interpretation = {
            "original_query": query,
            "matched_headers": [],
            "suggested_paths": [],
            "parent_child_relations": []
        }
        
        query_lower = query.lower()
        
        # Find matching headers
        for header, context in header_context.items():
            if header.lower() in query_lower:
                interpretation["matched_headers"].append(header)
                
                # Add parent-child context
                if context["parent"]:
                    interpretation["parent_child_relations"].append({
                        "child": header,
                        "parent": context["parent"]
                    })
                
                if context["children"]:
                    interpretation["parent_child_relations"].append({
                        "parent": header,
                        "children": context["children"]
                    })
        
        return interpretation


class MergedCellHandler:
    """
    Handles retrieval and interpretation of tables with merged cells.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_merged_cells(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process merged cells for improved retrieval.
        
        Args:
            table_data: Table data with merged cells
            
        Returns:
            Enhanced table data with merged cell processing
        """
        merged_cells = table_data.get("structure", {}).get("mergedCells", [])
        
        if not merged_cells:
            return table_data
        
        # Create merged cell index for fast lookup
        merged_index = self._create_merged_cell_index(merged_cells)
        
        # Propagate merged cell values
        cells = self._propagate_merged_values(
            table_data.get("cells", []),
            merged_cells
        )
        
        # Create span map for interpretation
        span_map = self._create_span_map(merged_cells)
        
        # Enhance table data
        enhanced_data = table_data.copy()
        enhanced_data["merged_cell_processing"] = {
            "merged_index": merged_index,
            "span_map": span_map,
            "total_merged": len(merged_cells),
            "max_row_span": max((m.get("row_span", 1) for m in merged_cells), default=1),
            "max_col_span": max((m.get("col_span", 1) for m in merged_cells), default=1)
        }
        enhanced_data["cells"] = cells
        
        return enhanced_data
    
    def _create_merged_cell_index(self, merged_cells: List[Dict]) -> Dict[Tuple[int, int], Dict]:
        """Create an index of merged cells by position."""
        index = {}
        
        for merged in merged_cells:
            start_row = merged.get("start_row", merged.get("row_index", 0))
            start_col = merged.get("start_col", merged.get("column_index", 0))
            row_span = merged.get("row_span", 1)
            col_span = merged.get("col_span", 1)
            
            # Mark all positions covered by this merged cell
            for r in range(start_row, start_row + row_span):
                for c in range(start_col, start_col + col_span):
                    index[(r, c)] = {
                        "origin": (start_row, start_col),
                        "content": merged.get("content", ""),
                        "is_origin": (r == start_row and c == start_col),
                        "row_span": row_span,
                        "col_span": col_span
                    }
        
        return index
    
    def _propagate_merged_values(self, cells: List[Dict], merged_cells: List[Dict]) -> List[Dict]:
        """Propagate merged cell values to all covered positions."""
        merged_index = self._create_merged_cell_index(merged_cells)
        enhanced_cells = []
        
        for cell in cells:
            enhanced_cell = cell.copy()
            pos = (cell.get("row", 0), cell.get("column", 0))
            
            if pos in merged_index:
                merge_info = merged_index[pos]
                if not merge_info["is_origin"] and not enhanced_cell.get("content"):
                    # Propagate content from origin cell
                    enhanced_cell["content"] = merge_info["content"]
                    enhanced_cell["is_merged_propagation"] = True
                
                enhanced_cell["merge_info"] = merge_info
            
            enhanced_cells.append(enhanced_cell)
        
        return enhanced_cells
    
    def _create_span_map(self, merged_cells: List[Dict]) -> Dict[str, List[Tuple[int, int]]]:
        """Create a map of content to all positions it spans."""
        span_map = {}
        
        for merged in merged_cells:
            content = merged.get("content", "")
            if content:
                start_row = merged.get("start_row", merged.get("row_index", 0))
                start_col = merged.get("start_col", merged.get("column_index", 0))
                row_span = merged.get("row_span", 1)
                col_span = merged.get("col_span", 1)
                
                positions = []
                for r in range(start_row, start_row + row_span):
                    for c in range(start_col, start_col + col_span):
                        positions.append((r, c))
                
                if content not in span_map:
                    span_map[content] = []
                span_map[content].extend(positions)
        
        return span_map
    
    def interpret_merged_cell_query(self, query: str, span_map: Dict) -> Dict[str, Any]:
        """
        Interpret query considering merged cells.
        
        Args:
            query: User query
            span_map: Map of content to spanned positions
            
        Returns:
            Query interpretation with merged cell context
        """
        interpretation = {
            "query": query,
            "matched_merged_content": [],
            "spanned_positions": []
        }
        
        query_lower = query.lower()
        
        for content, positions in span_map.items():
            if content.lower() in query_lower or query_lower in content.lower():
                interpretation["matched_merged_content"].append(content)
                interpretation["spanned_positions"].extend(positions)
        
        return interpretation


class BorderlessTableHandler:
    """
    Handles tables without clear borders or with implicit structure.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def process_borderless_table(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process borderless tables by inferring structure.
        
        Args:
            table_data: Table data potentially without clear borders
            
        Returns:
            Enhanced table data with inferred structure
        """
        # Check if table is borderless
        is_borderless = table_data.get("metadata", {}).get("is_borderless", False)
        
        if not is_borderless:
            # Try to detect if borderless
            is_borderless = self._detect_borderless_structure(table_data)
        
        if is_borderless:
            # Infer structure
            inferred_structure = self._infer_table_structure(table_data)
            
            # Enhance data
            enhanced_data = table_data.copy()
            enhanced_data["borderless_processing"] = {
                "is_borderless": True,
                "inferred_structure": inferred_structure,
                "confidence": inferred_structure.get("confidence", 0)
            }
            
            return enhanced_data
        
        return table_data
    
    def _detect_borderless_structure(self, table_data: Dict) -> bool:
        """Detect if table has borderless characteristics."""
        # Check metadata hints
        metadata = table_data.get("metadata", {})
        if "border" in str(metadata).lower() and "0" in str(metadata):
            return True
        
        # Check for spacing patterns in cells
        cells = table_data.get("cells", [])
        if cells:
            # Look for consistent spacing patterns
            positions = [(c.get("row", 0), c.get("column", 0)) for c in cells]
            if self._has_irregular_spacing(positions):
                return True
        
        return False
    
    def _has_irregular_spacing(self, positions: List[Tuple[int, int]]) -> bool:
        """Check if cell positions have irregular spacing."""
        if len(positions) < 2:
            return False
        
        # Check for gaps in positions
        rows = set(p[0] for p in positions)
        cols = set(p[1] for p in positions)
        
        if rows and cols:
            expected_cells = len(rows) * len(cols)
            actual_cells = len(positions)
            
            # If many cells are missing, likely borderless
            if actual_cells < expected_cells * 0.7:
                return True
        
        return False
    
    def _infer_table_structure(self, table_data: Dict) -> Dict[str, Any]:
        """Infer structure for borderless tables."""
        cells = table_data.get("cells", [])
        
        inferred = {
            "method": "spacing_analysis",
            "confidence": 0.5,
            "inferred_columns": [],
            "inferred_rows": [],
            "section_boundaries": []
        }
        
        if not cells:
            return inferred
        
        # Analyze content patterns
        content_patterns = self._analyze_content_patterns(cells)
        
        # Infer column boundaries based on content alignment
        column_groups = self._group_by_alignment(cells)
        inferred["inferred_columns"] = column_groups
        
        # Infer row groupings
        row_groups = self._identify_row_sections(cells)
        inferred["inferred_rows"] = row_groups
        
        # Calculate confidence
        if column_groups and row_groups:
            inferred["confidence"] = 0.8
        elif column_groups or row_groups:
            inferred["confidence"] = 0.6
        
        return inferred
    
    def _analyze_content_patterns(self, cells: List[Dict]) -> Dict[str, Any]:
        """Analyze content patterns in cells."""
        patterns = {
            "numeric_columns": [],
            "text_columns": [],
            "header_candidates": []
        }
        
        # Group cells by column
        by_column = {}
        for cell in cells:
            col = cell.get("column", 0)
            if col not in by_column:
                by_column[col] = []
            by_column[col].append(cell.get("content", ""))
        
        # Analyze each column
        for col, contents in by_column.items():
            numeric_count = sum(1 for c in contents if self._is_numeric(c))
            
            if numeric_count > len(contents) * 0.7:
                patterns["numeric_columns"].append(col)
            else:
                patterns["text_columns"].append(col)
        
        return patterns
    
    def _is_numeric(self, content: str) -> bool:
        """Check if content is numeric."""
        if not content:
            return False
        
        # Remove common numeric formatting
        cleaned = content.replace(',', '').replace('$', '').replace('%', '').strip()
        
        try:
            float(cleaned)
            return True
        except:
            return False
    
    def _group_by_alignment(self, cells: List[Dict]) -> List[Dict]:
        """Group cells by alignment to infer columns."""
        # Simplified grouping based on column position
        column_groups = {}
        
        for cell in cells:
            col = cell.get("column", 0)
            if col not in column_groups:
                column_groups[col] = {
                    "column_index": col,
                    "cells": [],
                    "dominant_type": "unknown"
                }
            column_groups[col]["cells"].append(cell)
        
        # Determine dominant type for each column
        for group in column_groups.values():
            numeric_count = sum(1 for c in group["cells"] if self._is_numeric(c.get("content", "")))
            if numeric_count > len(group["cells"]) * 0.5:
                group["dominant_type"] = "numeric"
            else:
                group["dominant_type"] = "text"
        
        return list(column_groups.values())
    
    def _identify_row_sections(self, cells: List[Dict]) -> List[Dict]:
        """Identify logical row sections."""
        row_sections = []
        
        # Group by row
        by_row = {}
        for cell in cells:
            row = cell.get("row", 0)
            if row not in by_row:
                by_row[row] = []
            by_row[row].append(cell)
        
        # Identify section breaks
        current_section = {"start_row": 0, "end_row": 0, "type": "data"}
        
        for row in sorted(by_row.keys()):
            row_cells = by_row[row]
            
            # Check if this row is a header or separator
            if self._is_header_row(row_cells):
                if current_section["end_row"] > current_section["start_row"]:
                    row_sections.append(current_section)
                current_section = {"start_row": row, "end_row": row, "type": "header"}
            else:
                if current_section["type"] == "header":
                    row_sections.append(current_section)
                    current_section = {"start_row": row, "end_row": row, "type": "data"}
                else:
                    current_section["end_row"] = row
        
        # Add last section
        if current_section["end_row"] >= current_section["start_row"]:
            row_sections.append(current_section)
        
        return row_sections
    
    def _is_header_row(self, row_cells: List[Dict]) -> bool:
        """Determine if a row is likely a header."""
        # Simple heuristic: mostly non-numeric content
        numeric_count = sum(1 for c in row_cells if self._is_numeric(c.get("content", "")))
        return numeric_count < len(row_cells) * 0.3


class ComplexTableInterpreter:
    """
    Main interpreter that combines all complex table handlers for optimal retrieval.
    """
    
    def __init__(self):
        self.hierarchical_processor = HierarchicalHeaderProcessor()
        self.merged_handler = MergedCellHandler()
        self.borderless_handler = BorderlessTableHandler()
        self.logger = logging.getLogger(__name__)
    
    def interpret_table_for_retrieval(self, table_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Interpret complex table for optimal retrieval.
        
        Args:
            table_data: Raw table data
            query: User query
            
        Returns:
            Interpreted table with enhanced retrieval information
        """
        # Determine complexity
        complexity = self._assess_complexity(table_data)
        
        # Process based on complexity
        interpreted = table_data.copy()
        interpreted["complexity_assessment"] = complexity.value
        
        # Apply appropriate processors
        if complexity in [TableComplexity.HIGH, TableComplexity.VERY_HIGH]:
            # Process hierarchical headers
            if self._has_hierarchical_headers(table_data):
                interpreted = self.hierarchical_processor.process_hierarchical_headers(interpreted)
            
            # Process merged cells
            if self._has_merged_cells(table_data):
                interpreted = self.merged_handler.process_merged_cells(interpreted)
            
            # Process borderless structure
            if self._is_borderless(table_data):
                interpreted = self.borderless_handler.process_borderless_table(interpreted)
        
        # Add query interpretation
        interpreted["query_interpretation"] = self._interpret_query_context(interpreted, query)
        
        return interpreted
    
    def _assess_complexity(self, table_data: Dict) -> TableComplexity:
        """Assess table complexity level."""
        score = 0
        
        features = table_data.get("complex_features", {})
        
        if features.get("has_merged_cells"):
            score += 2
        if features.get("has_hierarchical_headers"):
            score += 2
        if features.get("has_nested_structure"):
            score += 3
        if features.get("is_borderless"):
            score += 1
        
        # Check dimensions
        metadata = table_data.get("metadata", {})
        rows = metadata.get("row_count", 0)
        cols = metadata.get("column_count", 0)
        
        if rows > 50 or cols > 20:
            score += 2
        elif rows > 20 or cols > 10:
            score += 1
        
        if score >= 6:
            return TableComplexity.VERY_HIGH
        elif score >= 4:
            return TableComplexity.HIGH
        elif score >= 2:
            return TableComplexity.MEDIUM
        else:
            return TableComplexity.SIMPLE
    
    def _has_hierarchical_headers(self, table_data: Dict) -> bool:
        """Check if table has hierarchical headers."""
        return table_data.get("complex_features", {}).get("has_hierarchical_headers", False)
    
    def _has_merged_cells(self, table_data: Dict) -> bool:
        """Check if table has merged cells."""
        return table_data.get("complex_features", {}).get("has_merged_cells", False)
    
    def _is_borderless(self, table_data: Dict) -> bool:
        """Check if table is borderless."""
        return table_data.get("complex_features", {}).get("is_borderless", False)
    
    def _interpret_query_context(self, table_data: Dict, query: str) -> Dict[str, Any]:
        """Interpret query in context of table structure."""
        interpretation = {
            "query": query,
            "relevant_features": [],
            "suggested_search_paths": [],
            "warnings": []
        }
        
        # Check which features are relevant to the query
        if "hierarchical_headers" in table_data:
            header_interpretation = self.hierarchical_processor.interpret_hierarchical_query(
                query,
                table_data["hierarchical_headers"].get("context_map", {})
            )
            if header_interpretation["matched_headers"]:
                interpretation["relevant_features"].append("hierarchical_headers")
                interpretation["suggested_search_paths"].extend(
                    header_interpretation.get("suggested_paths", [])
                )
        
        if "merged_cell_processing" in table_data:
            merged_interpretation = self.merged_handler.interpret_merged_cell_query(
                query,
                table_data["merged_cell_processing"].get("span_map", {})
            )
            if merged_interpretation["matched_merged_content"]:
                interpretation["relevant_features"].append("merged_cells")
                interpretation["warnings"].append(
                    "Query matches merged cells - results span multiple positions"
                )
        
        # Add complexity warning if needed
        if table_data.get("complexity_assessment") in ["high", "very_high"]:
            interpretation["warnings"].append(
                f"Complex table structure ({table_data['complexity_assessment']}) - interpretation may require careful analysis"
            )
        
        return interpretation