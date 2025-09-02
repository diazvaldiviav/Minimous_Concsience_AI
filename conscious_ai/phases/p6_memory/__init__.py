"""
Phase 6: Memory Consolidation System
=====================================
Intelligent memory management with three-layer architecture:
- Working Memory: Recent interactions (10-15 items)
- Episodic Buffer: Compressed memories (30-40 items)
- Core Knowledge: Permanent facts (5-10 items)

Prevents unbounded memory growth while preserving critical information.
"""

from .memory_types import MemoryItem, MemoryType, MemoryLayers
from .memory_manager import MemoryManager
from .persistence import MemoryPersistence
from .memory_utils import (
    calculate_similarity,
    find_similar_memories,
    is_personal_info,
    is_preference,
    extract_entities,
    merge_memories,
    abstract_memories
)

__all__ = [
    'MemoryItem',
    'MemoryType', 
    'MemoryLayers',
    'MemoryManager',
    'MemoryPersistence',
    'calculate_similarity',
    'find_similar_memories',
    'is_personal_info',
    'is_preference',
    'extract_entities',
    'merge_memories',
    'abstract_memories'
]