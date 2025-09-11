"""
Test suite for BaseModelManager.

This module contains comprehensive tests for TinyLlama model loading,
tokenization, device management, and error handling.
"""

import asyncio
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from src.core.config import Settings
from src.core.exceptions import (
    ModelLoadError,
    ModelNotLoadedError,
    TokenizationError,
    ConfigurationError,
    MemoryError,
)
from src.memory.base_model import BaseModelManager, create_model_manager


class TestBaseModelManager:
    """Test cases for BaseModelManager."""
    
    @pytest.fixture
    def settings(self):
        """Create test settings."""
        return Settings(
            model=Settings.ModelConfig(
                name="test-model",
                cache_dir="./test_cache",
                device="cpu",
                max_length=512,
                torch_dtype="float32",
                low_cpu_mem_usage=True,
                trust_remote_code=False,
                use_safetensors=True
            )
        )
    
    @pytest.fixture
    def model_manager(self, settings):
        """Create BaseModelManager instance for testing."""
        return BaseModelManager(settings=settings)
    
    def test_init_default_settings(self):
        """Test initialization with default settings."""
        manager = BaseModelManager()
        assert manager._model_name is not None
        assert manager._device is not None
        assert not manager._is_loaded
        assert manager._model is None
        assert manager._tokenizer is None
    
    def test_init_custom_parameters(self, settings):
        """Test initialization with custom parameters."""
        manager = BaseModelManager(
            model_name="custom-model",
            device="cuda:0",
            settings=settings
        )
        assert manager._model_name == "custom-model"
        assert manager._device == "cuda:0"
        assert manager._settings == settings
    
    def test_device_detection_cpu(self, model_manager):
        """Test device detection defaults to CPU when CUDA unavailable."""
        with patch('torch.cuda.is_available', return_value=False):
            model_manager._detect_device()
            assert model_manager._resolved_device == "cpu"
            assert model_manager._device_type == "cpu"
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_name', return_value="Test GPU")
    def test_device_detection_cuda_auto(self, mock_name, mock_available, settings):
        """Test device detection with CUDA available and auto device."""
        manager = BaseModelManager(device="auto", settings=settings)
        assert manager._resolved_device == "cuda:0"
        assert manager._device_type == "cuda"
    
    def test_device_detection_explicit_device(self, settings):
        """Test device detection with explicit device specification."""
        manager = BaseModelManager(device="cuda:1", settings=settings)
        assert manager._resolved_device == "cuda:1"
        assert manager._device_type == "cuda"
    
    def test_get_model_config_cpu(self, model_manager):
        """Test model configuration for CPU device."""
        config = model_manager._get_model_config()
        
        assert "cache_dir" in config
        assert config["low_cpu_mem_usage"] is True
        assert config["trust_remote_code"] is False
        assert config["use_safetensors"] is True
        assert config["torch_dtype"] == torch.float32
        assert config["device_map"][""] == "cpu"
    
    @patch('torch.cuda.is_available', return_value=True)
    def test_get_model_config_cuda(self, mock_cuda, settings):
        """Test model configuration for CUDA device."""
        manager = BaseModelManager(device="cuda:0", settings=settings)
        config = manager._get_model_config()
        
        assert config["device_map"] == "auto"
        assert config["torch_dtype"] in [torch.float16, torch.float32]
    
    def test_get_torch_dtype_mapping(self, model_manager):
        """Test torch dtype mapping."""
        model_manager._settings.model.torch_dtype = "float32"
        assert model_manager._get_torch_dtype() == torch.float32
        
        model_manager._settings.model.torch_dtype = "float16"
        model_manager._device_type = "cpu"
        # Should fallback to float32 for CPU
        assert model_manager._get_torch_dtype() == torch.float32
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_should_use_quantization_low_memory(self, mock_props, model_manager):
        """Test quantization decision with low GPU memory."""
        # Mock GPU with 4GB memory
        mock_device = Mock()
        mock_device.total_memory = 4 * 1024 * 1024 * 1024  # 4GB in bytes
        mock_props.return_value = mock_device
        
        model_manager._device_type = "cuda"
        assert model_manager._should_use_quantization() is True
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.get_device_properties')
    def test_should_use_quantization_high_memory(self, mock_props, model_manager):
        """Test quantization decision with high GPU memory."""
        # Mock GPU with 24GB memory
        mock_device = Mock()
        mock_device.total_memory = 24 * 1024 * 1024 * 1024  # 24GB in bytes
        mock_props.return_value = mock_device
        
        model_manager._device_type = "cuda"
        assert model_manager._should_use_quantization() is False
    
    def test_should_use_quantization_cpu(self, model_manager):
        """Test quantization decision for CPU device."""
        model_manager._device_type = "cpu"
        assert model_manager._should_use_quantization() is False
    
    @patch('src.memory.base_model.AutoTokenizer')
    @patch('src.memory.base_model.AutoModelForCausalLM')
    @patch('src.memory.base_model.pipeline')
    @pytest.mark.asyncio
    async def test_load_model_success(self, mock_pipeline, mock_model_class, mock_tokenizer_class, model_manager):
        """Test successful model loading."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_pipeline_instance = Mock()
        mock_pipeline.return_value = mock_pipeline_instance
        
        # Load model
        await model_manager.load_model()
        
        # Verify state
        assert model_manager._is_loaded is True
        assert model_manager._tokenizer == mock_tokenizer
        assert model_manager._model == mock_model
        assert model_manager._pipeline == mock_pipeline_instance
        
        # Verify pad token was set
        assert mock_tokenizer.pad_token == "</s>"
    
    @patch('src.memory.base_model.AutoTokenizer')
    @pytest.mark.asyncio
    async def test_load_model_tokenizer_failure(self, mock_tokenizer_class, model_manager):
        """Test model loading failure during tokenizer loading."""
        mock_tokenizer_class.from_pretrained.side_effect = Exception("Tokenizer load failed")
        
        with pytest.raises(ModelLoadError) as exc_info:
            await model_manager.load_model()
        
        assert "Tokenizer load failed" in str(exc_info.value)
        assert not model_manager._is_loaded
        assert model_manager._model is None
        assert model_manager._tokenizer is None
    
    @patch('src.memory.base_model.AutoTokenizer')
    @patch('src.memory.base_model.AutoModelForCausalLM')
    @pytest.mark.asyncio
    async def test_load_model_memory_error(self, mock_model_class, mock_tokenizer_class, model_manager):
        """Test model loading failure due to memory error."""
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.side_effect = RuntimeError("CUDA out of memory")
        
        with pytest.raises(MemoryError) as exc_info:
            await model_manager.load_model()
        
        assert "Insufficient memory" in str(exc_info.value)
        assert not model_manager._is_loaded
    
    @patch('src.memory.base_model.AutoTokenizer')
    @patch('src.memory.base_model.AutoModelForCausalLM')
    @pytest.mark.asyncio
    async def test_load_model_already_loaded(self, mock_model_class, mock_tokenizer_class, model_manager):
        """Test loading model when already loaded."""
        model_manager._is_loaded = True
        
        await model_manager.load_model()
        
        # Should not call loading functions
        mock_tokenizer_class.from_pretrained.assert_not_called()
        mock_model_class.from_pretrained.assert_not_called()
    
    def test_tokenize_not_loaded(self, model_manager):
        """Test tokenization when model not loaded."""
        with pytest.raises(ModelNotLoadedError):
            model_manager.tokenize("test text")
    
    @patch('src.memory.base_model.AutoTokenizer')
    @patch('src.memory.base_model.AutoModelForCausalLM')
    @patch('src.memory.base_model.pipeline')
    @pytest.mark.asyncio
    async def test_tokenize_success(self, mock_pipeline, mock_model_class, mock_tokenizer_class, model_manager):
        """Test successful tokenization."""
        # Setup loaded model
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        # Setup tokenizer return
        expected_tokens = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }
        mock_tokenizer.return_value = expected_tokens
        
        await model_manager.load_model()
        
        # Test tokenization
        result = model_manager.tokenize("test text", max_length=10)
        
        # Verify tokenizer was called correctly
        mock_tokenizer.assert_called_once_with(
            "test text",
            max_length=10,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        assert result == expected_tokens
    
    def test_tokenize_invalid_input(self, model_manager):
        """Test tokenization with invalid input types."""
        model_manager._is_loaded = True
        model_manager._tokenizer = Mock()
        
        # Test non-string input
        with pytest.raises(TokenizationError) as exc_info:
            model_manager.tokenize(123)
        assert "must be a string" in str(exc_info.value)
        
        # Test empty string
        with pytest.raises(TokenizationError) as exc_info:
            model_manager.tokenize("")
        assert "cannot be empty" in str(exc_info.value)
        
        with pytest.raises(TokenizationError) as exc_info:
            model_manager.tokenize("   ")
        assert "cannot be empty" in str(exc_info.value)
    
    @patch('src.memory.base_model.AutoTokenizer')
    @patch('src.memory.base_model.AutoModelForCausalLM')
    @patch('src.memory.base_model.pipeline')
    @pytest.mark.asyncio
    async def test_tokenize_device_placement(self, mock_pipeline, mock_model_class, mock_tokenizer_class, settings):
        """Test tokenization with CUDA device placement."""
        # Create manager with CUDA device
        manager = BaseModelManager(device="cuda:0", settings=settings)
        
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "</s>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        # Setup tokenizer return with mock tensors
        mock_tensor = Mock()
        mock_tensor.to.return_value = mock_tensor
        expected_tokens = {
            'input_ids': mock_tensor,
            'attention_mask': mock_tensor
        }
        mock_tokenizer.return_value = expected_tokens
        
        with patch('torch.cuda.is_available', return_value=True):
            await manager.load_model()
            result = manager.tokenize("test text")
            
            # Verify tensors were moved to device
            mock_tensor.to.assert_called_with("cuda:0")
    
    @patch('src.memory.base_model.AutoTokenizer')
    @patch('src.memory.base_model.AutoModelForCausalLM')
    @patch('src.memory.base_model.pipeline')
    @pytest.mark.asyncio
    async def test_get_model_info(self, mock_pipeline, mock_model_class, mock_tokenizer_class, model_manager):
        """Test getting model information."""
        # Setup loaded model with parameters
        mock_model = Mock()
        mock_param = Mock()
        mock_param.numel.return_value = 1000000  # 1M parameters
        mock_model.parameters.return_value = [mock_param, mock_param]  # 2M total
        
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_pipeline.return_value = Mock()
        
        await model_manager.load_model()
        
        info = model_manager.get_model_info()
        
        assert info.name == model_manager._model_name
        assert info.type == "language_model"
        assert info.device == model_manager._resolved_device
        assert info.is_loaded is True
        assert info.parameters_count == 2000000
        assert info.load_time is not None
        assert "torch_dtype" in info.metadata
        assert "max_length" in info.metadata
    
    def test_get_model_info_not_loaded(self, model_manager):
        """Test getting model info when not loaded."""
        info = model_manager.get_model_info()
        
        assert info.name == model_manager._model_name
        assert info.type == "language_model"
        assert info.is_loaded is False
        assert info.parameters_count is None
        assert info.load_time is None
    
    def test_is_loaded(self, model_manager):
        """Test is_loaded method."""
        assert model_manager.is_loaded() is False
        
        # Simulate loaded state
        model_manager._is_loaded = True
        model_manager._model = Mock()
        model_manager._tokenizer = Mock()
        
        assert model_manager.is_loaded() is True
    
    def test_get_device(self, model_manager):
        """Test get_device method."""
        device = model_manager.get_device()
        assert device == model_manager._resolved_device
    
    def test_get_tokenizer_not_loaded(self, model_manager):
        """Test getting tokenizer when not loaded."""
        with pytest.raises(ModelNotLoadedError):
            model_manager.get_tokenizer()
    
    @patch('src.memory.base_model.AutoTokenizer')
    @patch('src.memory.base_model.AutoModelForCausalLM')
    @patch('src.memory.base_model.pipeline')
    @pytest.mark.asyncio
    async def test_get_tokenizer_loaded(self, mock_pipeline, mock_model_class, mock_tokenizer_class, model_manager):
        """Test getting tokenizer when loaded."""
        mock_tokenizer = Mock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        await model_manager.load_model()
        
        result = model_manager.get_tokenizer()
        assert result == mock_tokenizer
    
    def test_get_model_not_loaded(self, model_manager):
        """Test getting model when not loaded."""
        with pytest.raises(ModelNotLoadedError):
            model_manager.get_model()
    
    @patch('src.memory.base_model.AutoTokenizer')
    @patch('src.memory.base_model.AutoModelForCausalLM')
    @patch('src.memory.base_model.pipeline')
    @pytest.mark.asyncio
    async def test_get_model_loaded(self, mock_pipeline, mock_model_class, mock_tokenizer_class, model_manager):
        """Test getting model when loaded."""
        mock_model = Mock()
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_pipeline.return_value = Mock()
        
        await model_manager.load_model()
        
        result = model_manager.get_model()
        assert result == mock_model
    
    @patch('src.memory.base_model.AutoTokenizer')
    @patch('src.memory.base_model.AutoModelForCausalLM')
    @patch('src.memory.base_model.pipeline')
    @patch('torch.cuda.empty_cache')
    @patch('gc.collect')
    @pytest.mark.asyncio
    async def test_unload_model(self, mock_gc, mock_cuda_cache, mock_pipeline, mock_model_class, mock_tokenizer_class, model_manager):
        """Test model unloading."""
        # Load model first
        mock_tokenizer_class.from_pretrained.return_value = Mock()
        mock_model_class.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        await model_manager.load_model()
        assert model_manager.is_loaded()
        
        # Unload model
        model_manager.unload_model()
        
        assert not model_manager.is_loaded()
        assert model_manager._model is None
        assert model_manager._tokenizer is None
        assert model_manager._pipeline is None
        mock_gc.assert_called_once()
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    @patch('gc.collect')
    def test_unload_model_cuda_cleanup(self, mock_gc, mock_cuda_cache, settings):
        """Test model unloading with CUDA cleanup."""
        manager = BaseModelManager(device="cuda:0", settings=settings)
        manager._is_loaded = True
        manager._model = Mock()
        manager._tokenizer = Mock()
        
        manager.unload_model()
        
        mock_cuda_cache.assert_called_once()
        mock_gc.assert_called_once()


class TestCreateModelManager:
    """Test cases for create_model_manager function."""
    
    def test_create_model_manager_defaults(self):
        """Test creating model manager with defaults."""
        manager = create_model_manager()
        assert isinstance(manager, BaseModelManager)
        assert manager._model_name is not None
        assert manager._device is not None
    
    def test_create_model_manager_custom(self):
        """Test creating model manager with custom parameters."""
        settings = Settings()
        manager = create_model_manager(
            model_name="custom-model",
            device="cpu",
            settings=settings
        )
        
        assert isinstance(manager, BaseModelManager)
        assert manager._model_name == "custom-model"
        assert manager._device == "cpu"
        assert manager._settings == settings


# Performance and integration tests
class TestBaseModelManagerIntegration:
    """Integration tests for BaseModelManager."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    @patch('src.memory.base_model.AutoTokenizer')
    @patch('src.memory.base_model.AutoModelForCausalLM')
    @patch('src.memory.base_model.pipeline')
    @pytest.mark.asyncio
    async def test_full_model_lifecycle(self, mock_pipeline, mock_model_class, mock_tokenizer_class):
        """Test complete model lifecycle from load to unload."""
        # Setup mocks for realistic behavior
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3]]),
            'attention_mask': torch.tensor([[1, 1, 1]])
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_param = Mock()
        mock_param.numel.return_value = 500000
        mock_model.parameters.return_value = [mock_param] * 4  # 2M parameters
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_pipeline.return_value = Mock()
        
        # Create manager
        settings = Settings(model=Settings.ModelConfig(device="cpu", max_length=128))
        manager = BaseModelManager(model_name="test/model", settings=settings)
        
        # Test lifecycle
        assert not manager.is_loaded()
        
        # Load model
        await manager.load_model()
        assert manager.is_loaded()
        
        # Test operations
        info = manager.get_model_info()
        assert info.is_loaded
        assert info.parameters_count == 2000000
        
        # Test tokenization
        tokens = manager.tokenize("Hello world!")
        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        
        # Test model/tokenizer access
        tokenizer = manager.get_tokenizer()
        model = manager.get_model()
        assert tokenizer == mock_tokenizer
        assert model == mock_model
        
        # Unload model
        manager.unload_model()
        assert not manager.is_loaded()
        
        # Verify cleanup
        with pytest.raises(ModelNotLoadedError):
            manager.get_model()
        with pytest.raises(ModelNotLoadedError):
            manager.get_tokenizer()
    
    @pytest.mark.performance
    @patch('src.memory.base_model.AutoTokenizer')
    @patch('src.memory.base_model.AutoModelForCausalLM')
    @patch('src.memory.base_model.pipeline')
    @pytest.mark.asyncio
    async def test_tokenization_performance(self, mock_pipeline, mock_model_class, mock_tokenizer_class):
        """Test tokenization performance with various input sizes."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "</s>"
        
        def mock_tokenize_call(*args, **kwargs):
            text = args[0]
            max_length = kwargs.get('max_length', 512)
            # Simulate realistic token counts
            token_count = min(len(text.split()), max_length)
            return {
                'input_ids': torch.randint(0, 1000, (1, token_count)),
                'attention_mask': torch.ones(1, token_count)
            }
        
        mock_tokenizer.side_effect = mock_tokenize_call
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = Mock()
        mock_pipeline.return_value = Mock()
        
        manager = BaseModelManager()
        await manager.load_model()
        
        # Test with different text lengths
        test_texts = [
            "Short text",
            " ".join(["word"] * 50),  # Medium text
            " ".join(["word"] * 200),  # Long text
        ]
        
        for text in test_texts:
            start_time = asyncio.get_event_loop().time()
            result = manager.tokenize(text)
            end_time = asyncio.get_event_loop().time()
            
            # Verify result format
            assert "input_ids" in result
            assert "attention_mask" in result
            
            # Performance should be reasonable (< 100ms for mocked tokenization)
            duration = end_time - start_time
            assert duration < 0.1, f"Tokenization took too long: {duration}s for text length {len(text)}"


if __name__ == "__main__":
    pytest.main([__file__])