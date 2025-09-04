"""
Memory Utility Functions for Phase 6
====================================
Helper functions for memory operations including similarity calculation,
classification heuristics, and compression functions.
"""

import re
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2')
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDER = None
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using fallback similarity")


def calculate_similarity(mem1: 'MemoryItem', mem2: 'MemoryItem') -> float:
    """
    Calculate cosine similarity between two memory items.
    Falls back to text-based similarity if embeddings not available.
    """
    if EMBEDDINGS_AVAILABLE and mem1.embedding and mem2.embedding:
        # Use cosine similarity on embeddings
        vec1 = np.array(mem1.embedding)
        vec2 = np.array(mem2.embedding)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    else:
        # Fallback: Jaccard similarity on words
        words1 = set(mem1.content.lower().split())
        words2 = set(mem2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0


def find_similar_memories(memory: 'MemoryItem', memory_list: List['MemoryItem'], 
                         threshold: float = 0.85) -> List[Tuple['MemoryItem', float]]:
    """
    Find memories similar to the given memory above threshold.
    Returns list of (memory, similarity_score) tuples.
    """
    similar = []
    for other in memory_list:
        if other != memory:
            similarity = calculate_similarity(memory, other)
            if similarity >= threshold:
                similar.append((other, similarity))
    
    return sorted(similar, key=lambda x: x[1], reverse=True)


def is_personal_info(text: str) -> bool:
    """
    Detect if text contains personal information like names, relationships, personal data.
    """
    personal_patterns = [
        r'\bmy name is\b',
        r'\bi am\b',
        r'\bi\'m\b',
        r'\buser\'?s? name\b',
        r'\bcall me\b',
        r'\bmy (wife|husband|partner|child|children|family)\b',
        r'\bi live\b',
        r'\bmy (age|birthday|job|work|profession)\b',
        r'\b(email|phone|address)\b',
        r'\bremember (that |me)\b'
    ]
    
    text_lower = text.lower()
    for pattern in personal_patterns:
        if re.search(pattern, text_lower):
            return True
    
    # Check for proper nouns (capitalized words that might be names)
    words = text.split()
    capitalized = [w for w in words if w and w[0].isupper() and w.lower() not in ['i']]
    if len(capitalized) >= 1 and any(len(w) > 2 for w in capitalized):
        # Likely contains names
        return True
    
    return False


def is_preference(text: str) -> bool:
    """
    Identify if text contains likes, dislikes, opinions, or preferences.
    """
    preference_patterns = [
        r'\bi (like|love|prefer|enjoy|hate|dislike)\b',
        r'\bmy favorite\b',
        r'\bi don\'?t like\b',
        r'\bi always\b',
        r'\bi never\b',
        r'\bi think\b',
        r'\bi believe\b',
        r'\bi feel\b',
        r'\bi want\b',
        r'\bi need\b',
        r'\bprefer(ence)?\b',
        r'\b(good|bad|better|worse|best|worst)\b',
        r'\b(should|shouldn\'?t)\b'
    ]
    
    text_lower = text.lower()
    for pattern in preference_patterns:
        if re.search(pattern, text_lower):
            return True
    
    return False


def extract_entities(text: str) -> List[str]:
    """
    Extract important nouns and concepts from text.
    Simple implementation using capitalization and common patterns.
    """
    entities = []
    
    # Extract capitalized words (potential names/places)
    words = text.split()
    for i, word in enumerate(words):
        # Skip first word of sentence
        if word and word[0].isupper() and (i == 0 or words[i-1][-1] not in '.!?'):
            clean_word = re.sub(r'[^\w\s]', '', word)
            if len(clean_word) > 2 and clean_word.lower() not in ['the', 'and', 'but', 'for']:
                entities.append(clean_word)
    
    # Extract quoted text
    quoted = re.findall(r'"([^"]+)"', text)
    entities.extend(quoted)
    
    # Extract numbers with context
    numbers_with_context = re.findall(r'\b\d+\s+\w+\b', text)
    entities.extend(numbers_with_context)
    
    return list(set(entities))  # Remove duplicates


def merge_memories(memory_list: List['MemoryItem']) -> 'MemoryItem':
    """
    Combine similar memories into a single summary memory.
    Preserves the most important information from all memories.
    """
    if not memory_list:
        return None
    
    if len(memory_list) == 1:
        return memory_list[0]
    
    # Sort by relevance to prioritize important content
    sorted_memories = sorted(memory_list, key=lambda m: m.relevance, reverse=True)
    
    # Extract key information from each memory
    key_phrases = []
    all_entities = []
    max_relevance = 0.0
    total_access = 0
    
    for mem in sorted_memories:
        # Extract entities from each memory
        entities = extract_entities(mem.content)
        all_entities.extend(entities)
        
        # Keep important phrases
        if is_personal_info(mem.content) or is_preference(mem.content):
            key_phrases.append(mem.content)
        elif len(key_phrases) < 3:  # Limit to avoid too long summaries
            # Take first sentence or up to 50 chars
            sentence = mem.content.split('.')[0]
            if len(sentence) > 50:
                sentence = sentence[:50] + "..."
            key_phrases.append(sentence)
        
        max_relevance = max(max_relevance, mem.relevance)
        total_access += mem.access_count
    
    # Create merged content
    unique_entities = list(set(all_entities))
    
    if key_phrases:
        merged_content = "Summary: " + " | ".join(key_phrases[:3])
        if unique_entities:
            merged_content += f" [Entities: {', '.join(unique_entities[:5])}]"
    else:
        merged_content = f"Compressed memory of {len(memory_list)} similar items"
        if unique_entities:
            merged_content += f" about: {', '.join(unique_entities[:5])}"
    
    # Create new memory item
    from .memory_types import MemoryItem, MemoryType
    
    merged = MemoryItem(
        content=merged_content,
        relevance=max_relevance * 0.9,  # Slight decay for compression
        timestamp=datetime.now(),
        access_count=total_access,
        decay_rate=0.05,  # Slower decay for compressed memories
        memory_type=MemoryType.EPISODIC,
        metadata={
            'compressed_from': len(memory_list),
            'original_timestamps': [m.timestamp.isoformat() for m in memory_list]
        }
    )
    
    # Generate embedding if available
    if EMBEDDINGS_AVAILABLE and EMBEDDER:
        try:
            merged.embedding = EMBEDDER.encode(merged_content).tolist()
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
    
    return merged


def abstract_memories(specific_list: List['MemoryItem']) -> 'MemoryItem':
    """
    Generate general/abstract memory from specific instances.
    Useful for creating rules or patterns from examples.
    """
    if not specific_list:
        return None
    
    # Analyze patterns across memories
    common_entities = {}
    common_actions = {}
    preferences = []
    
    for mem in specific_list:
        # Count entity occurrences
        entities = extract_entities(mem.content)
        for entity in entities:
            common_entities[entity] = common_entities.get(entity, 0) + 1
        
        # Detect preferences
        if is_preference(mem.content):
            preferences.append(mem.content)
        
        # Extract verbs/actions (simple approach)
        verbs = re.findall(r'\b(is|are|was|were|like|prefer|want|need|have|has|do|does)\b', 
                          mem.content.lower())
        for verb in verbs:
            common_actions[verb] = common_actions.get(verb, 0) + 1
    
    # Generate abstract content
    abstract_content = "Pattern recognized: "
    
    # Add most common entities
    if common_entities:
        top_entities = sorted(common_entities.items(), key=lambda x: x[1], reverse=True)[:3]
        abstract_content += f"Frequently discusses {', '.join([e[0] for e in top_entities])}. "
    
    # Add preference patterns
    if preferences:
        if len(preferences) > 2:
            abstract_content += f"Has expressed {len(preferences)} preferences/opinions. "
        else:
            abstract_content += "Has some preferences. "
    
    # Calculate average relevance
    avg_relevance = sum(m.relevance for m in specific_list) / len(specific_list)
    
    # Create abstract memory
    from .memory_types import MemoryItem, MemoryType
    
    abstract = MemoryItem(
        content=abstract_content,
        relevance=min(1.0, avg_relevance * 1.1),  # Boost for abstraction
        timestamp=datetime.now(),
        access_count=sum(m.access_count for m in specific_list),
        decay_rate=0.02,  # Very slow decay for abstract knowledge
        memory_type=MemoryType.EPISODIC,
        metadata={
            'abstracted_from': len(specific_list),
            'abstraction_type': 'pattern',
            'confidence': min(1.0, len(specific_list) / 5.0)  # Higher confidence with more examples
        }
    )
    
    # Generate embedding if available
    if EMBEDDINGS_AVAILABLE and EMBEDDER:
        try:
            abstract.embedding = EMBEDDER.encode(abstract_content).tolist()
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
    
    return abstract


def get_embedding(text: str) -> Optional[List[float]]:
    """
    Generate embedding for text if sentence-transformers available.
    """
    if EMBEDDINGS_AVAILABLE and EMBEDDER:
        try:
            return EMBEDDER.encode(text).tolist()
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
    return None