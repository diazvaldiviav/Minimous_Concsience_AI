"""
Memory Data Structures for Phase 6
===================================
Defines the three-layer memory architecture with appropriate data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import json


class MemoryType(Enum):
    """Types of memory in the three-layer architecture"""
    WORKING = "working"      # Short-term, recent interactions
    EPISODIC = "episodic"    # Medium-term, compressed memories
    CORE = "core"            # Long-term, permanent knowledge


@dataclass
class MemoryItem:
    """
    Individual memory item with metadata for intelligent management.
    """
    content: str                              # The actual memory content
    relevance: float                          # Current relevance score (0.0-1.0)
    timestamp: datetime                       # When created
    access_count: int = 0                     # Times referenced
    decay_rate: float = 0.1                   # How quickly relevance decreases
    memory_type: MemoryType = MemoryType.WORKING  # Current layer
    embedding: Optional[List[float]] = None   # Semantic embedding for similarity
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def decay(self) -> None:
        """Apply decay to relevance score"""
        self.relevance = max(0.0, self.relevance - self.decay_rate)
    
    def access(self) -> None:
        """Record access and boost relevance"""
        self.access_count += 1
        self.relevance = min(1.0, self.relevance + 0.1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'content': self.content,
            'relevance': self.relevance,
            'timestamp': self.timestamp.isoformat(),
            'access_count': self.access_count,
            'decay_rate': self.decay_rate,
            'memory_type': self.memory_type.value,
            'embedding': self.embedding,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create from dictionary (for JSON deserialization)"""
        return cls(
            content=data['content'],
            relevance=data['relevance'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            access_count=data.get('access_count', 0),
            decay_rate=data.get('decay_rate', 0.1),
            memory_type=MemoryType(data.get('memory_type', 'working')),
            embedding=data.get('embedding'),
            metadata=data.get('metadata', {})
        )
    
    def __repr__(self) -> str:
        return f"MemoryItem(type={self.memory_type.value}, relevance={self.relevance:.2f}, content='{self.content[:50]}...')"


class MemoryLayers:
    """
    Three-layer memory architecture managing distinct collections.
    """
    
    # Capacity limits for each layer
    WORKING_CAPACITY = 15      # Recent interactions
    EPISODIC_CAPACITY = 40     # Compressed memories
    CORE_CAPACITY = 10         # Permanent facts
    
    def __init__(self):
        """Initialize the three memory layers"""
        self.working_memory: List[MemoryItem] = []
        self.episodic_buffer: List[MemoryItem] = []
        self.core_knowledge: List[MemoryItem] = []
        
        # Track consolidation statistics
        self.stats = {
            'total_memories_created': 0,
            'consolidations_performed': 0,
            'memories_compressed': 0,
            'memories_promoted': 0,
            'last_consolidation': None
        }
    
    def add_to_working(self, item: MemoryItem) -> bool:
        """
        Add memory to working layer with FIFO policy.
        Returns True if successful, False if consolidation needed.
        """
        item.memory_type = MemoryType.WORKING
        self.working_memory.append(item)
        self.stats['total_memories_created'] += 1
        
        if len(self.working_memory) > self.WORKING_CAPACITY:
            return False  # Needs consolidation
        return True
    
    def add_to_episodic(self, item: MemoryItem) -> None:
        """Add memory to episodic buffer"""
        item.memory_type = MemoryType.EPISODIC
        self.episodic_buffer.append(item)
        
        # Enforce capacity limit (remove lowest relevance)
        if len(self.episodic_buffer) > self.EPISODIC_CAPACITY:
            self.episodic_buffer.sort(key=lambda x: x.relevance)
            self.episodic_buffer.pop(0)
    
    def add_to_core(self, item: MemoryItem) -> None:
        """Add memory to core knowledge (permanent)"""
        item.memory_type = MemoryType.CORE
        item.decay_rate = 0.0  # Core memories don't decay
        self.core_knowledge.append(item)
        
        # Enforce capacity limit (keep most accessed)
        if len(self.core_knowledge) > self.CORE_CAPACITY:
            self.core_knowledge.sort(key=lambda x: x.access_count)
            self.core_knowledge.pop(0)
    
    def get_all_memories(self) -> List[MemoryItem]:
        """Get all memories across layers"""
        return self.working_memory + self.episodic_buffer + self.core_knowledge
    
    def get_layer(self, memory_type: MemoryType) -> List[MemoryItem]:
        """Get memories from specific layer"""
        if memory_type == MemoryType.WORKING:
            return self.working_memory
        elif memory_type == MemoryType.EPISODIC:
            return self.episodic_buffer
        elif memory_type == MemoryType.CORE:
            return self.core_knowledge
        else:
            return []
    
    def clear_layer(self, memory_type: MemoryType) -> int:
        """Clear specific memory layer, returns number of items cleared"""
        if memory_type == MemoryType.WORKING:
            count = len(self.working_memory)
            self.working_memory.clear()
            return count
        elif memory_type == MemoryType.EPISODIC:
            count = len(self.episodic_buffer)
            self.episodic_buffer.clear()
            return count
        elif memory_type == MemoryType.CORE:
            count = len(self.core_knowledge)
            self.core_knowledge.clear()
            return count
        return 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current memory distribution and statistics"""
        return {
            'working_memory': {
                'count': len(self.working_memory),
                'capacity': self.WORKING_CAPACITY,
                'usage': len(self.working_memory) / self.WORKING_CAPACITY
            },
            'episodic_buffer': {
                'count': len(self.episodic_buffer),
                'capacity': self.EPISODIC_CAPACITY,
                'usage': len(self.episodic_buffer) / self.EPISODIC_CAPACITY
            },
            'core_knowledge': {
                'count': len(self.core_knowledge),
                'capacity': self.CORE_CAPACITY,
                'usage': len(self.core_knowledge) / self.CORE_CAPACITY
            },
            'total_memories': len(self.get_all_memories()),
            'total_capacity': self.WORKING_CAPACITY + self.EPISODIC_CAPACITY + self.CORE_CAPACITY,
            'consolidation_stats': self.stats
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'working_memory': [m.to_dict() for m in self.working_memory],
            'episodic_buffer': [m.to_dict() for m in self.episodic_buffer],
            'core_knowledge': [m.to_dict() for m in self.core_knowledge],
            'stats': self.stats
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryLayers':
        """Create from dictionary (for JSON deserialization)"""
        layers = cls()
        
        # Load memories
        layers.working_memory = [
            MemoryItem.from_dict(m) for m in data.get('working_memory', [])
        ]
        layers.episodic_buffer = [
            MemoryItem.from_dict(m) for m in data.get('episodic_buffer', [])
        ]
        layers.core_knowledge = [
            MemoryItem.from_dict(m) for m in data.get('core_knowledge', [])
        ]
        
        # Load stats
        layers.stats = data.get('stats', layers.stats)
        
        return layers
    
    def __repr__(self) -> str:
        return (f"MemoryLayers(working={len(self.working_memory)}/{self.WORKING_CAPACITY}, "
                f"episodic={len(self.episodic_buffer)}/{self.EPISODIC_CAPACITY}, "
                f"core={len(self.core_knowledge)}/{self.CORE_CAPACITY})")