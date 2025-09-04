"""
Memory Manager for Phase 6
===========================
Central memory management system with intelligent consolidation.
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .memory_types import MemoryItem, MemoryType, MemoryLayers
from .memory_utils import (
    calculate_similarity,
    find_similar_memories,
    is_personal_info,
    is_preference,
    extract_entities,
    merge_memories,
    abstract_memories,
    get_embedding
)

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Central memory management system with three-layer architecture
    and intelligent consolidation algorithms.
    """
    
    def __init__(self, auto_persist: bool = True, persistence_path: Optional[str] = None):
        """
        Initialize the memory manager.
        
        Args:
            auto_persist: Whether to automatically save to disk
            persistence_path: Custom path for memory storage
        """
        self.memory_layers = MemoryLayers()
        self.auto_persist = auto_persist
        self.persistence_path = persistence_path
        
        # Consolidation tracking
        self.cycles_since_consolidation = 0
        self.consolidation_in_progress = False
        
        # Performance metrics
        self.last_consolidation_time = None
        self.consolidation_duration_ms = 0
        
        # Load existing memories if persistence enabled
        if self.auto_persist:
            self._load_memories()
    
    def add_memory(self, content: str, relevance: float = 0.5, 
                   auto_consolidate: bool = True) -> MemoryItem:
        """
        Add new memory to working layer.
        
        Args:
            content: The memory content
            relevance: Initial relevance score
            auto_consolidate: Whether to trigger consolidation if needed
        
        Returns:
            The created MemoryItem
        """
        # Create new memory item
        memory = MemoryItem(
            content=content,
            relevance=min(1.0, max(0.0, relevance)),
            timestamp=datetime.now(),
            memory_type=MemoryType.WORKING
        )
        
        # Generate embedding if available
        embedding = get_embedding(content)
        if embedding:
            memory.embedding = embedding
        
        # Check if this is important information
        if is_personal_info(content):
            memory.relevance = min(1.0, memory.relevance + 0.3)
            memory.metadata['is_personal'] = True
            
            # Personal info might go directly to core
            if "name is" in content.lower() or "call me" in content.lower():
                memory.memory_type = MemoryType.CORE
                memory.decay_rate = 0.0
                self.memory_layers.add_to_core(memory)
                logger.info(f"Added personal info directly to core: {content[:50]}...")
                self._persist_if_enabled()
                return memory
        
        if is_preference(content):
            memory.relevance = min(1.0, memory.relevance + 0.2)
            memory.metadata['is_preference'] = True
        
        # Add to working memory
        success = self.memory_layers.add_to_working(memory)
        
        # Check if consolidation needed
        if not success and auto_consolidate:
            logger.info("Working memory full, triggering consolidation")
            self.consolidate()
            # Try adding again after consolidation
            self.memory_layers.add_to_working(memory)
        
        self._persist_if_enabled()
        return memory
    
    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[Tuple[MemoryItem, float]]:
        """
        Get relevant memories across all layers.
        
        Args:
            query: The search query
            top_k: Number of memories to return
        
        Returns:
            List of (memory, relevance_score) tuples
        """
        results = []
        
        # Create query embedding if available
        query_embedding = get_embedding(query)
        
        # Search across all layers
        all_memories = self.memory_layers.get_all_memories()
        
        # Debug: Log retrieval attempt
        logger.warning(f"🔍 RETRIEVAL: Searching for '{query[:50]}...' in {len(all_memories)} total memories")
        
        # Debug: Show memory distribution
        working_count = len(self.memory_layers.working_memory)
        episodic_count = len(self.memory_layers.episodic_buffer)
        core_count = len(self.memory_layers.core_knowledge)
        logger.warning(f"🔍 RETRIEVAL: Memory distribution - Working: {working_count}, Episodic: {episodic_count}, Core: {core_count}")
        
        for memory in all_memories:
            # Calculate relevance score
            relevance = 0.0
            
            # Similarity to query
            if query_embedding and memory.embedding:
                # Use embedding similarity
                query_vec = np.array(query_embedding)
                mem_vec = np.array(memory.embedding)
                
                dot_product = np.dot(query_vec, mem_vec)
                norm1 = np.linalg.norm(query_vec)
                norm2 = np.linalg.norm(mem_vec)
                
                if norm1 > 0 and norm2 > 0:
                    similarity = dot_product / (norm1 * norm2)
                    relevance = similarity * 0.6  # Weight similarity
            else:
                # Fallback to text similarity
                query_words = set(query.lower().split())
                mem_words = set(memory.content.lower().split())
                
                if query_words and mem_words:
                    intersection = query_words.intersection(mem_words)
                    similarity = len(intersection) / len(query_words)
                    relevance = similarity * 0.6
            
            # Factor in memory's inherent relevance
            relevance += memory.relevance * 0.3
            
            # Boost for core memories
            if memory.memory_type == MemoryType.CORE:
                relevance += 0.1
            
            # Record access
            memory.access()
            
            results.append((memory, relevance))
        
        # Sort by relevance and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Debug: Log retrieval results
        top_results = results[:top_k]
        logger.warning(f"🔍 RETRIEVAL RESULT: Found {len(top_results)} relevant memories")
        if top_results:
            for i, (mem, score) in enumerate(top_results[:3], 1):
                logger.warning(f"🔍 RETRIEVAL RESULT {i}: Score={score:.3f}, Content='{mem.content[:100]}...'")
        
        self._persist_if_enabled()
        return top_results
    
    def consolidate(self) -> Dict[str, Any]:
        """
        Trigger memory consolidation process.
        
        Returns:
            Statistics about the consolidation
        """
        if self.consolidation_in_progress:
            logger.warning("Consolidation already in progress")
            return {'status': 'in_progress'}
        
        self.consolidation_in_progress = True
        start_time = time.time()
        stats = {
            'memories_compressed': 0,
            'memories_promoted': 0,
            'memories_decayed': 0,
            'memories_removed': 0
        }
        
        try:
            # Step 1: Apply decay to all memories
            for memory in self.memory_layers.get_all_memories():
                if memory.memory_type != MemoryType.CORE:  # Core memories don't decay
                    old_relevance = memory.relevance
                    memory.decay()
                    if old_relevance > 0 and memory.relevance <= 0:
                        stats['memories_decayed'] += 1
            
            # Step 2: Remove irrelevant memories (relevance <= 0)
            for layer_type in [MemoryType.WORKING, MemoryType.EPISODIC]:
                layer = self.memory_layers.get_layer(layer_type)
                initial_count = len(layer)
                layer[:] = [m for m in layer if m.relevance > 0]
                stats['memories_removed'] += initial_count - len(layer)
            
            # Step 3: Identify and compress similar memories in working memory
            working = self.memory_layers.working_memory.copy()
            compressed_groups = []
            processed = set()
            
            for i, memory in enumerate(working):
                if i in processed:
                    continue
                
                # Find similar memories
                similar = find_similar_memories(memory, working[i+1:], threshold=0.85)
                if similar:
                    group = [memory] + [m for m, _ in similar]
                    compressed_groups.append(group)
                    
                    # Mark as processed
                    processed.add(i)
                    for other, _ in similar:
                        idx = working.index(other)
                        processed.add(idx)
            
            # Compress similar groups
            for group in compressed_groups:
                if len(group) >= 2:
                    merged = merge_memories(group)
                    if merged:
                        self.memory_layers.add_to_episodic(merged)
                        stats['memories_compressed'] += len(group)
                        
                        # Remove originals from working memory
                        for mem in group:
                            if mem in self.memory_layers.working_memory:
                                self.memory_layers.working_memory.remove(mem)
            
            # Step 4: Promote important working memories to episodic
            working_to_promote = []
            for memory in self.memory_layers.working_memory:
                # Promote if: high relevance, personal info, or frequently accessed
                if (memory.relevance >= 0.7 or 
                    memory.metadata.get('is_personal', False) or
                    memory.metadata.get('is_preference', False) or
                    memory.access_count >= 3):
                    working_to_promote.append(memory)
            
            for memory in working_to_promote:
                self.memory_layers.working_memory.remove(memory)
                self.memory_layers.add_to_episodic(memory)
                stats['memories_promoted'] += 1
            
            # Step 5: Check episodic buffer for core promotion
            episodic_to_core = []
            for memory in self.memory_layers.episodic_buffer:
                # Promote to core if: very high relevance, personal info, or very frequently accessed
                if (memory.relevance >= 0.9 or
                    (memory.metadata.get('is_personal', True) and memory.relevance >= 0.7) or
                    memory.access_count >= 10):
                    episodic_to_core.append(memory)
            
            for memory in episodic_to_core[:3]:  # Limit promotions per consolidation
                self.memory_layers.episodic_buffer.remove(memory)
                self.memory_layers.add_to_core(memory)
                stats['memories_promoted'] += 1
            
            # Step 6: Generate abstractions if we have enough similar memories
            if len(self.memory_layers.episodic_buffer) >= 10:
                # Look for patterns to abstract
                episodic = self.memory_layers.episodic_buffer
                preference_memories = [m for m in episodic if m.metadata.get('is_preference', False)]
                
                if len(preference_memories) >= 3:
                    abstract = abstract_memories(preference_memories)
                    if abstract:
                        self.memory_layers.add_to_episodic(abstract)
                        logger.info(f"Generated abstraction: {abstract.content[:100]}")
            
            # Update consolidation tracking
            self.memory_layers.stats['consolidations_performed'] += 1
            self.memory_layers.stats['memories_compressed'] += stats['memories_compressed']
            self.memory_layers.stats['memories_promoted'] += stats['memories_promoted']
            self.memory_layers.stats['last_consolidation'] = datetime.now().isoformat()
            
            self.cycles_since_consolidation = 0
            self.last_consolidation_time = datetime.now()
            
        finally:
            self.consolidation_in_progress = False
            self.consolidation_duration_ms = (time.time() - start_time) * 1000
            stats['duration_ms'] = self.consolidation_duration_ms
            
            # Persist after consolidation
            self._persist_if_enabled()
            
            logger.info(f"Consolidation complete: {stats}")
        
        return stats
    
    def should_consolidate(self) -> bool:
        """
        Check if consolidation is needed.
        
        Returns:
            True if consolidation should be triggered
        """
        # Check working memory capacity
        if len(self.memory_layers.working_memory) > 15:
            return True
        
        # Check cycles since last consolidation
        if self.cycles_since_consolidation >= 5:
            return True
        
        # Check total memory usage
        total = len(self.memory_layers.get_all_memories())
        if total > 50:  # Approaching max capacity
            return True
        
        return False
    
    def increment_cycle(self) -> None:
        """Increment the cycle counter for consolidation tracking"""
        self.cycles_since_consolidation += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        stats = self.memory_layers.get_statistics()
        
        # Add manager-level stats
        stats['manager'] = {
            'cycles_since_consolidation': self.cycles_since_consolidation,
            'last_consolidation_time': self.last_consolidation_time.isoformat() if self.last_consolidation_time else None,
            'last_consolidation_duration_ms': self.consolidation_duration_ms,
            'auto_persist': self.auto_persist,
            'should_consolidate': self.should_consolidate()
        }
        
        return stats
    
    def clear_layer(self, layer_type: MemoryType) -> int:
        """
        Clear a specific memory layer.
        
        Args:
            layer_type: Which layer to clear
        
        Returns:
            Number of memories cleared
        """
        count = self.memory_layers.clear_layer(layer_type)
        self._persist_if_enabled()
        logger.info(f"Cleared {count} memories from {layer_type.value} layer")
        return count
    
    def _persist_if_enabled(self) -> None:
        """Persist memories to disk if auto_persist is enabled"""
        if self.auto_persist:
            try:
                from .persistence import MemoryPersistence
                persistence = MemoryPersistence()
                
                path = self.persistence_path or persistence.default_path
                persistence.save_to_json(self.memory_layers, path)
            except Exception as e:
                logger.error(f"Failed to persist memories: {e}")
    
    def _load_memories(self) -> None:
        """Load memories from disk if they exist"""
        try:
            from .persistence import MemoryPersistence
            persistence = MemoryPersistence()
            
            path = self.persistence_path or persistence.default_path
            loaded = persistence.load_from_json(path)
            
            if loaded:
                self.memory_layers = loaded
                logger.info(f"Loaded {len(self.memory_layers.get_all_memories())} memories from disk")
        except FileNotFoundError:
            logger.info("No existing memories found, starting fresh")
        except Exception as e:
            logger.error(f"Failed to load memories: {e}")
    
    def save_memories(self) -> bool:
        """
        Manually save memories to disk.
        
        Returns:
            True if successful
        """
        try:
            from .persistence import MemoryPersistence
            persistence = MemoryPersistence()
            
            path = self.persistence_path or persistence.default_path
            persistence.save_to_json(self.memory_layers, path)
            logger.info(f"Manually saved memories to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save memories: {e}")
            return False