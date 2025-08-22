"""
Table Chunking Strategy Module
Handles chunking of large tables for Azure AI Search retrieval optimization
Focus: Maintaining table coherence while respecting Azure Search limits
"""
import json
import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkingMethod(Enum):
    """Methods for chunking tables."""
    ROW_BASED = "row_based"
    COLUMN_BASED = "column_based"
    SECTION_BASED = "section_based"
    DELIMITER_ENCODING = "delimiter_encoding"


@dataclass
class TableChunk:
    """Represents a chunk of a table."""
    chunk_id: str
    parent_id: str
    chunk_index: int
    total_chunks: int
    method: ChunkingMethod
    data: Dict[str, Any]
    metadata: Dict[str, Any]


class TableChunkingStrategy:
    """
    Implements intelligent chunking strategies for large tables to optimize retrieval
    while respecting Azure AI Search limits (3000 elements per collection).
    """
    
    # Azure AI Search limits
    MAX_COLLECTION_ELEMENTS = 3000
    SAFE_CELL_LIMIT = 2500  # Leave margin for metadata
    MAX_CHUNK_SIZE_KB = 32  # Max size per field in KB
    
    def __init__(self, max_cells_per_chunk: int = None):
        """
        Initialize chunking strategy.
        
        Args:
            max_cells_per_chunk: Maximum cells per chunk (default: SAFE_CELL_LIMIT)
        """
        self.max_cells_per_chunk = max_cells_per_chunk or self.SAFE_CELL_LIMIT
        self.logger = logging.getLogger(__name__)
    
    def chunk_table_for_retrieval(self, table_data: Dict[str, Any]) -> List[TableChunk]:
        """
        Chunk table data for optimal retrieval.
        
        Args:
            table_data: Complete table data
            
        Returns:
            List of table chunks optimized for retrieval
        """
        # Assess table size and structure
        assessment = self._assess_table_for_chunking(table_data)
        
        if not assessment["needs_chunking"]:
            # Return single chunk
            return [self._create_single_chunk(table_data)]
        
        # Choose chunking method based on table characteristics
        method = self._select_chunking_method(assessment)
        
        # Apply chunking method
        if method == ChunkingMethod.DELIMITER_ENCODING:
            return self._chunk_with_delimiter_encoding(table_data)
        elif method == ChunkingMethod.SECTION_BASED:
            return self._chunk_by_sections(table_data)
        elif method == ChunkingMethod.COLUMN_BASED:
            return self._chunk_by_columns(table_data)
        else:  # ROW_BASED
            return self._chunk_by_rows(table_data)
    
    def _assess_table_for_chunking(self, table_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess if and how table needs chunking."""
        cells = table_data.get("cells", [])
        structure = table_data.get("structure", {})
        
        assessment = {
            "total_cells": len(cells),
            "needs_chunking": len(cells) > self.max_cells_per_chunk,
            "has_sections": False,
            "has_merged_cells": bool(structure.get("mergedCells", [])),
            "has_hierarchical_headers": bool(structure.get("headers", {}).get("hierarchicalMap", [])),
            "estimated_chunks": 0
        }
        
        if assessment["needs_chunking"]:
            assessment["estimated_chunks"] = (len(cells) + self.max_cells_per_chunk - 1) // self.max_cells_per_chunk
        
        # Check for logical sections
        if "inferred_rows" in table_data.get("borderless_processing", {}):
            assessment["has_sections"] = True
        
        return assessment
    
    def _select_chunking_method(self, assessment: Dict[str, Any]) -> ChunkingMethod:
        """Select optimal chunking method based on table characteristics."""
        # If table has logical sections, use section-based chunking
        if assessment["has_sections"]:
            return ChunkingMethod.SECTION_BASED
        
        # If table has many cells but simple structure, use delimiter encoding
        if assessment["total_cells"] > self.MAX_COLLECTION_ELEMENTS * 2:
            return ChunkingMethod.DELIMITER_ENCODING
        
        # If table has merged cells or hierarchical headers, preserve rows
        if assessment["has_merged_cells"] or assessment["has_hierarchical_headers"]:
            return ChunkingMethod.ROW_BASED
        
        # Default to row-based chunking
        return ChunkingMethod.ROW_BASED
    
    def _create_single_chunk(self, table_data: Dict[str, Any]) -> TableChunk:
        """Create a single chunk for small tables."""
        chunk_id = self._generate_chunk_id(table_data.get("table_id", ""), 0)
        
        return TableChunk(
            chunk_id=chunk_id,
            parent_id=table_data.get("table_id", chunk_id),
            chunk_index=0,
            total_chunks=1,
            method=ChunkingMethod.ROW_BASED,
            data=table_data,
            metadata={
                "is_complete": True,
                "row_range": [0, table_data.get("metadata", {}).get("row_count", 0)],
                "column_range": [0, table_data.get("metadata", {}).get("column_count", 0)]
            }
        )
    
    def _chunk_by_rows(self, table_data: Dict[str, Any]) -> List[TableChunk]:
        """Chunk table by rows, preserving column structure."""
        chunks = []
        cells = table_data.get("cells", [])
        
        # Sort cells by row
        cells_by_row = {}
        for cell in cells:
            row = cell.get("row", 0)
            if row not in cells_by_row:
                cells_by_row[row] = []
            cells_by_row[row].append(cell)
        
        # Create chunks
        current_chunk_cells = []
        chunk_index = 0
        start_row = 0
        
        for row in sorted(cells_by_row.keys()):
            row_cells = cells_by_row[row]
            
            # Check if adding this row would exceed limit
            if current_chunk_cells and len(current_chunk_cells) + len(row_cells) > self.max_cells_per_chunk:
                # Create chunk
                chunk = self._create_row_chunk(
                    table_data,
                    current_chunk_cells,
                    chunk_index,
                    start_row,
                    row - 1
                )
                chunks.append(chunk)
                
                # Reset for next chunk
                current_chunk_cells = row_cells
                chunk_index += 1
                start_row = row
            else:
                current_chunk_cells.extend(row_cells)
        
        # Create final chunk
        if current_chunk_cells:
            chunk = self._create_row_chunk(
                table_data,
                current_chunk_cells,
                chunk_index,
                start_row,
                max(cells_by_row.keys())
            )
            chunks.append(chunk)
        
        # Update total chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _create_row_chunk(self, table_data: Dict, cells: List[Dict], 
                         index: int, start_row: int, end_row: int) -> TableChunk:
        """Create a chunk from row-based data."""
        parent_id = table_data.get("table_id", "")
        chunk_id = self._generate_chunk_id(parent_id, index)
        
        # Create chunk data
        chunk_data = {
            **table_data,
            "cells": cells,
            "chunk_info": {
                "method": "row_based",
                "row_range": [start_row, end_row],
                "is_partial": True
            }
        }
        
        # Preserve headers in each chunk for context
        if "structure" in table_data and "headers" in table_data["structure"]:
            chunk_data["structure"]["headers"] = table_data["structure"]["headers"]
        
        return TableChunk(
            chunk_id=chunk_id,
            parent_id=parent_id,
            chunk_index=index,
            total_chunks=0,  # Will be updated
            method=ChunkingMethod.ROW_BASED,
            data=chunk_data,
            metadata={
                "row_range": [start_row, end_row],
                "cell_count": len(cells),
                "preserves_headers": True
            }
        )
    
    def _chunk_by_columns(self, table_data: Dict[str, Any]) -> List[TableChunk]:
        """Chunk table by columns, useful for wide tables."""
        chunks = []
        cells = table_data.get("cells", [])
        
        # Sort cells by column
        cells_by_column = {}
        for cell in cells:
            col = cell.get("column", 0)
            if col not in cells_by_column:
                cells_by_column[col] = []
            cells_by_column[col].append(cell)
        
        # Group columns into chunks
        current_chunk_cells = []
        chunk_index = 0
        start_col = 0
        column_group = []
        
        for col in sorted(cells_by_column.keys()):
            col_cells = cells_by_column[col]
            
            if current_chunk_cells and len(current_chunk_cells) + len(col_cells) > self.max_cells_per_chunk:
                # Create chunk
                chunk = self._create_column_chunk(
                    table_data,
                    current_chunk_cells,
                    chunk_index,
                    column_group
                )
                chunks.append(chunk)
                
                # Reset
                current_chunk_cells = col_cells
                column_group = [col]
                chunk_index += 1
            else:
                current_chunk_cells.extend(col_cells)
                column_group.append(col)
        
        # Final chunk
        if current_chunk_cells:
            chunk = self._create_column_chunk(
                table_data,
                current_chunk_cells,
                chunk_index,
                column_group
            )
            chunks.append(chunk)
        
        # Update total chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _create_column_chunk(self, table_data: Dict, cells: List[Dict], 
                            index: int, columns: List[int]) -> TableChunk:
        """Create a chunk from column-based data."""
        parent_id = table_data.get("table_id", "")
        chunk_id = self._generate_chunk_id(parent_id, index)
        
        chunk_data = {
            **table_data,
            "cells": cells,
            "chunk_info": {
                "method": "column_based",
                "column_indices": columns,
                "is_partial": True
            }
        }
        
        return TableChunk(
            chunk_id=chunk_id,
            parent_id=parent_id,
            chunk_index=index,
            total_chunks=0,
            method=ChunkingMethod.COLUMN_BASED,
            data=chunk_data,
            metadata={
                "column_indices": columns,
                "cell_count": len(cells)
            }
        )
    
    def _chunk_by_sections(self, table_data: Dict[str, Any]) -> List[TableChunk]:
        """Chunk table by logical sections (e.g., header sections, data sections)."""
        chunks = []
        cells = table_data.get("cells", [])
        
        # Get section information
        sections = table_data.get("borderless_processing", {}).get("inferred_structure", {}).get("inferred_rows", [])
        
        if not sections:
            # Fallback to row-based chunking
            return self._chunk_by_rows(table_data)
        
        for idx, section in enumerate(sections):
            section_cells = [
                cell for cell in cells
                if section["start_row"] <= cell.get("row", 0) <= section["end_row"]
            ]
            
            if len(section_cells) > self.max_cells_per_chunk:
                # Section too large, need to sub-chunk
                sub_chunks = self._sub_chunk_section(table_data, section_cells, idx)
                chunks.extend(sub_chunks)
            else:
                chunk = self._create_section_chunk(table_data, section_cells, idx, section)
                chunks.append(chunk)
        
        # Update total chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _create_section_chunk(self, table_data: Dict, cells: List[Dict], 
                             index: int, section: Dict) -> TableChunk:
        """Create a chunk from a logical section."""
        parent_id = table_data.get("table_id", "")
        chunk_id = self._generate_chunk_id(parent_id, index)
        
        chunk_data = {
            **table_data,
            "cells": cells,
            "chunk_info": {
                "method": "section_based",
                "section_type": section.get("type", "data"),
                "section_range": [section["start_row"], section["end_row"]]
            }
        }
        
        return TableChunk(
            chunk_id=chunk_id,
            parent_id=parent_id,
            chunk_index=index,
            total_chunks=0,
            method=ChunkingMethod.SECTION_BASED,
            data=chunk_data,
            metadata={
                "section_type": section.get("type", "data"),
                "row_range": [section["start_row"], section["end_row"]],
                "cell_count": len(cells)
            }
        )
    
    def _sub_chunk_section(self, table_data: Dict, section_cells: List[Dict], 
                          section_index: int) -> List[TableChunk]:
        """Sub-chunk a large section."""
        sub_chunks = []
        
        for i in range(0, len(section_cells), self.max_cells_per_chunk):
            chunk_cells = section_cells[i:i + self.max_cells_per_chunk]
            parent_id = table_data.get("table_id", "")
            chunk_id = self._generate_chunk_id(parent_id, f"{section_index}_{i // self.max_cells_per_chunk}")
            
            chunk_data = {
                **table_data,
                "cells": chunk_cells,
                "chunk_info": {
                    "method": "section_based",
                    "is_sub_chunk": True,
                    "parent_section": section_index
                }
            }
            
            chunk = TableChunk(
                chunk_id=chunk_id,
                parent_id=parent_id,
                chunk_index=len(sub_chunks),
                total_chunks=0,
                method=ChunkingMethod.SECTION_BASED,
                data=chunk_data,
                metadata={
                    "is_sub_chunk": True,
                    "cell_count": len(chunk_cells)
                }
            )
            sub_chunks.append(chunk)
        
        return sub_chunks
    
    def _chunk_with_delimiter_encoding(self, table_data: Dict[str, Any]) -> List[TableChunk]:
        """
        Use delimiter encoding for very large tables.
        Encodes cells as delimited strings to maximize data per chunk.
        """
        cells = table_data.get("cells", [])
        encoded_cells = []
        
        # Encode cells as delimited strings
        for cell in cells:
            encoded = self._encode_cell(cell)
            encoded_cells.append(encoded)
        
        # Group encoded cells into chunks
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        
        for encoded in encoded_cells:
            cell_size = len(encoded.encode('utf-8'))
            
            # Check size limit (32KB per field)
            if current_chunk and current_size + cell_size > self.MAX_CHUNK_SIZE_KB * 1024:
                # Create chunk
                chunk = self._create_encoded_chunk(table_data, current_chunk, chunk_index)
                chunks.append(chunk)
                
                # Reset
                current_chunk = [encoded]
                current_size = cell_size
                chunk_index += 1
            else:
                current_chunk.append(encoded)
                current_size += cell_size
        
        # Final chunk
        if current_chunk:
            chunk = self._create_encoded_chunk(table_data, current_chunk, chunk_index)
            chunks.append(chunk)
        
        # Update total chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _encode_cell(self, cell: Dict) -> str:
        """Encode cell as delimited string."""
        # Format: |row|col|value|type|headers|
        row = cell.get("row", 0)
        col = cell.get("column", 0)
        value = cell.get("content", "").replace("|", "\\|")  # Escape delimiter
        cell_type = cell.get("kind", "content")
        
        # Encode headers if present
        headers = cell.get("headers", {})
        header_str = ""
        if headers:
            row_headers = ",".join(headers.get("row", []))
            col_headers = ",".join(headers.get("column", []))
            header_str = f"{row_headers};{col_headers}"
        
        return f"|{row}|{col}|{value}|{cell_type}|{header_str}|"
    
    def _create_encoded_chunk(self, table_data: Dict, encoded_cells: List[str], 
                             index: int) -> TableChunk:
        """Create chunk with delimiter-encoded cells."""
        parent_id = table_data.get("table_id", "")
        chunk_id = self._generate_chunk_id(parent_id, index)
        
        chunk_data = {
            "table_id": parent_id,
            "encoded_cells": encoded_cells,
            "encoding_method": "delimiter",
            "delimiter": "|",
            "structure": table_data.get("structure", {}),
            "metadata": table_data.get("metadata", {})
        }
        
        return TableChunk(
            chunk_id=chunk_id,
            parent_id=parent_id,
            chunk_index=index,
            total_chunks=0,
            method=ChunkingMethod.DELIMITER_ENCODING,
            data=chunk_data,
            metadata={
                "encoding": "delimiter",
                "cell_count": len(encoded_cells),
                "compressed": True
            }
        )
    
    def _generate_chunk_id(self, parent_id: str, index: Any) -> str:
        """Generate unique chunk ID."""
        return f"{parent_id}_chunk_{index}"
    
    def reconstruct_table_from_chunks(self, chunks: List[TableChunk]) -> Dict[str, Any]:
        """
        Reconstruct complete table from chunks for processing.
        
        Args:
            chunks: List of table chunks
            
        Returns:
            Reconstructed table data
        """
        if not chunks:
            return {}
        
        if len(chunks) == 1:
            return chunks[0].data
        
        # Sort chunks by index
        sorted_chunks = sorted(chunks, key=lambda x: x.chunk_index)
        
        # Determine reconstruction method
        method = sorted_chunks[0].method
        
        if method == ChunkingMethod.DELIMITER_ENCODING:
            return self._reconstruct_from_encoded_chunks(sorted_chunks)
        else:
            return self._reconstruct_from_cell_chunks(sorted_chunks)
    
    def _reconstruct_from_cell_chunks(self, chunks: List[TableChunk]) -> Dict[str, Any]:
        """Reconstruct table from cell-based chunks."""
        # Start with first chunk as base
        reconstructed = chunks[0].data.copy()
        all_cells = []
        
        for chunk in chunks:
            cells = chunk.data.get("cells", [])
            all_cells.extend(cells)
        
        # Remove duplicates based on (row, column)
        seen = set()
        unique_cells = []
        for cell in all_cells:
            key = (cell.get("row", 0), cell.get("column", 0))
            if key not in seen:
                seen.add(key)
                unique_cells.append(cell)
        
        reconstructed["cells"] = unique_cells
        reconstructed["is_reconstructed"] = True
        reconstructed["chunk_count"] = len(chunks)
        
        return reconstructed
    
    def _reconstruct_from_encoded_chunks(self, chunks: List[TableChunk]) -> Dict[str, Any]:
        """Reconstruct table from delimiter-encoded chunks."""
        all_encoded = []
        
        for chunk in chunks:
            encoded_cells = chunk.data.get("encoded_cells", [])
            all_encoded.extend(encoded_cells)
        
        # Decode cells
        cells = []
        for encoded in all_encoded:
            cell = self._decode_cell(encoded)
            if cell:
                cells.append(cell)
        
        # Reconstruct structure
        reconstructed = {
            "table_id": chunks[0].parent_id,
            "cells": cells,
            "structure": chunks[0].data.get("structure", {}),
            "metadata": chunks[0].data.get("metadata", {}),
            "is_reconstructed": True,
            "chunk_count": len(chunks)
        }
        
        return reconstructed
    
    def _decode_cell(self, encoded: str) -> Optional[Dict]:
        """Decode delimiter-encoded cell."""
        parts = encoded.strip("|").split("|")
        
        if len(parts) < 5:
            return None
        
        try:
            cell = {
                "row": int(parts[0]),
                "column": int(parts[1]),
                "content": parts[2].replace("\\|", "|"),  # Unescape delimiter
                "kind": parts[3]
            }
            
            # Decode headers if present
            if parts[4]:
                header_parts = parts[4].split(";")
                if len(header_parts) == 2:
                    cell["headers"] = {
                        "row": header_parts[0].split(",") if header_parts[0] else [],
                        "column": header_parts[1].split(",") if header_parts[1] else []
                    }
            
            return cell
        except Exception as e:
            self.logger.error(f"Error decoding cell: {e}")
            return None