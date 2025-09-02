"""
Memory Persistence Layer for Phase 6
====================================
Simple but robust JSON storage for memory persistence.
"""

import json
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import threading

from .memory_types import MemoryLayers

logger = logging.getLogger(__name__)


class MemoryPersistence:
    """
    Handles JSON persistence for memory storage with safeguards.
    """
    
    MAX_FILE_SIZE_MB = 5
    MAX_BACKUPS = 3
    
    def __init__(self, base_path: str = "./data/memories"):
        """
        Initialize persistence layer.
        
        Args:
            base_path: Base directory for memory storage
        """
        self.base_path = Path(base_path)
        self.default_path = self.base_path / "memory_store.json"
        
        # Thread lock for atomic operations
        self.lock = threading.Lock()
        
        # Ensure directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def save_to_json(self, memory_layers: MemoryLayers, filepath: Optional[Path] = None) -> bool:
        """
        Atomic save with backup rotation.
        
        Args:
            memory_layers: The memory layers to save
            filepath: Optional custom filepath
        
        Returns:
            True if successful
        """
        filepath = Path(filepath) if filepath else self.default_path
        
        with self.lock:
            try:
                # Check size before saving
                data = memory_layers.to_dict()
                json_str = json.dumps(data, indent=2)
                size_mb = len(json_str.encode()) / (1024 * 1024)
                
                if size_mb > self.MAX_FILE_SIZE_MB:
                    logger.warning(f"Memory file size ({size_mb:.2f}MB) exceeds limit ({self.MAX_FILE_SIZE_MB}MB)")
                    # Trigger aggressive consolidation
                    logger.info("Triggering emergency consolidation due to file size")
                    return False
                
                # Create backup if file exists
                if filepath.exists():
                    self._create_backup(filepath)
                
                # Write to temporary file first (atomic operation)
                temp_path = filepath.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                
                # Atomic rename
                temp_path.replace(filepath)
                
                logger.debug(f"Saved {len(memory_layers.get_all_memories())} memories to {filepath}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to save memories: {e}")
                # Try to restore from backup if save failed
                if filepath.exists():
                    backup_path = filepath.with_suffix('.backup')
                    if backup_path.exists():
                        shutil.copy2(backup_path, filepath)
                        logger.info("Restored from backup after save failure")
                return False
    
    def load_from_json(self, filepath: Optional[Path] = None) -> Optional[MemoryLayers]:
        """
        Load memories with validation and migration.
        
        Args:
            filepath: Optional custom filepath
        
        Returns:
            MemoryLayers object or None if failed
        """
        filepath = Path(filepath) if filepath else self.default_path
        
        if not filepath.exists():
            logger.debug(f"No memory file found at {filepath}")
            return None
        
        with self.lock:
            try:
                # Check file size
                size_mb = filepath.stat().st_size / (1024 * 1024)
                if size_mb > self.MAX_FILE_SIZE_MB * 1.5:  # Allow some overflow for loading
                    logger.warning(f"Memory file too large ({size_mb:.2f}MB), attempting recovery")
                    return self._recover_from_large_file(filepath)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Validate and migrate if needed
                data = self._validate_and_migrate(data)
                
                # Create MemoryLayers from data
                memory_layers = MemoryLayers.from_dict(data)
                
                logger.info(f"Loaded {len(memory_layers.get_all_memories())} memories from {filepath}")
                return memory_layers
                
            except json.JSONDecodeError as e:
                logger.error(f"Corrupted memory file: {e}")
                return self._recover_from_corruption(filepath)
            except Exception as e:
                logger.error(f"Failed to load memories: {e}")
                return None
    
    def rotate_backups(self) -> None:
        """Keep only the last MAX_BACKUPS backup files"""
        backup_pattern = self.default_path.stem + "_backup_*.json"
        backups = sorted(self.base_path.glob(backup_pattern))
        
        if len(backups) > self.MAX_BACKUPS:
            # Remove oldest backups
            for backup in backups[:-self.MAX_BACKUPS]:
                try:
                    backup.unlink()
                    logger.debug(f"Removed old backup: {backup}")
                except Exception as e:
                    logger.warning(f"Failed to remove backup {backup}: {e}")
    
    def _create_backup(self, filepath: Path) -> None:
        """Create a backup of the current file"""
        try:
            # Simple backup (overwrite)
            backup_path = filepath.with_suffix('.backup')
            shutil.copy2(filepath, backup_path)
            
            # Timestamped backup (for rotation)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_backup = filepath.parent / f"{filepath.stem}_backup_{timestamp}.json"
            shutil.copy2(filepath, timestamped_backup)
            
            # Rotate old backups
            self.rotate_backups()
            
            logger.debug(f"Created backup at {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to create backup: {e}")
    
    def _validate_and_migrate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data structure and migrate if needed.
        
        Args:
            data: Raw data from JSON
        
        Returns:
            Validated/migrated data
        """
        # Ensure required fields exist
        if 'working_memory' not in data:
            data['working_memory'] = []
        if 'episodic_buffer' not in data:
            data['episodic_buffer'] = []
        if 'core_knowledge' not in data:
            data['core_knowledge'] = []
        if 'stats' not in data:
            data['stats'] = {
                'total_memories_created': 0,
                'consolidations_performed': 0,
                'memories_compressed': 0,
                'memories_promoted': 0,
                'last_consolidation': None
            }
        
        # Validate memory items
        for layer_name in ['working_memory', 'episodic_buffer', 'core_knowledge']:
            validated_items = []
            for item in data[layer_name]:
                if self._validate_memory_item(item):
                    validated_items.append(item)
                else:
                    logger.warning(f"Skipping invalid memory item in {layer_name}")
            data[layer_name] = validated_items
        
        return data
    
    def _validate_memory_item(self, item: Dict[str, Any]) -> bool:
        """Validate a single memory item"""
        required_fields = ['content', 'relevance', 'timestamp']
        for field in required_fields:
            if field not in item:
                return False
        
        # Validate types
        if not isinstance(item['content'], str):
            return False
        if not isinstance(item['relevance'], (int, float)):
            return False
        
        return True
    
    def _recover_from_corruption(self, filepath: Path) -> Optional[MemoryLayers]:
        """
        Attempt to recover from a corrupted file.
        
        Args:
            filepath: Path to corrupted file
        
        Returns:
            Recovered MemoryLayers or None
        """
        logger.info("Attempting recovery from corrupted file")
        
        # Try backup
        backup_path = filepath.with_suffix('.backup')
        if backup_path.exists():
            logger.info("Loading from backup file")
            try:
                with open(backup_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data = self._validate_and_migrate(data)
                return MemoryLayers.from_dict(data)
            except Exception as e:
                logger.error(f"Backup also corrupted: {e}")
        
        # Try timestamped backups
        backup_pattern = filepath.stem + "_backup_*.json"
        backups = sorted(self.base_path.glob(backup_pattern), reverse=True)
        
        for backup in backups:
            try:
                logger.info(f"Trying backup: {backup}")
                with open(backup, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                data = self._validate_and_migrate(data)
                return MemoryLayers.from_dict(data)
            except Exception as e:
                logger.warning(f"Backup {backup} failed: {e}")
                continue
        
        logger.error("All recovery attempts failed")
        return None
    
    def _recover_from_large_file(self, filepath: Path) -> Optional[MemoryLayers]:
        """
        Recover from a file that's too large.
        
        Args:
            filepath: Path to large file
        
        Returns:
            Truncated MemoryLayers or None
        """
        logger.info("Attempting to recover from oversized file")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data = self._validate_and_migrate(data)
            
            # Truncate memories to fit size limit
            # Keep core knowledge, recent episodic, and minimal working
            if 'core_knowledge' in data:
                data['core_knowledge'] = data['core_knowledge'][:10]
            if 'episodic_buffer' in data:
                data['episodic_buffer'] = data['episodic_buffer'][-20:]  # Keep most recent
            if 'working_memory' in data:
                data['working_memory'] = data['working_memory'][-5:]  # Keep most recent
            
            memory_layers = MemoryLayers.from_dict(data)
            
            # Save truncated version
            self.save_to_json(memory_layers, filepath)
            
            logger.info(f"Recovered with {len(memory_layers.get_all_memories())} memories")
            return memory_layers
            
        except Exception as e:
            logger.error(f"Failed to recover from large file: {e}")
            return None
    
    def get_file_info(self, filepath: Optional[Path] = None) -> Dict[str, Any]:
        """
        Get information about the memory file.
        
        Args:
            filepath: Optional custom filepath
        
        Returns:
            Dictionary with file information
        """
        filepath = Path(filepath) if filepath else self.default_path
        
        if not filepath.exists():
            return {
                'exists': False,
                'path': str(filepath)
            }
        
        stat = filepath.stat()
        return {
            'exists': True,
            'path': str(filepath),
            'size_mb': stat.st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'backups': len(list(self.base_path.glob(filepath.stem + "_backup_*.json")))
        }