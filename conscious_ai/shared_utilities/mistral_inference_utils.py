"""
Shared utilities for Mistral model inference across the conscious AI system.
Provides consistent interface for Mistral model operations.
"""

import json
import re
import logging
import torch
from typing import Dict, List, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def format_mistral_prompt(system_message: str, user_message: str) -> str:
    """
    Format prompt in Mistral chat format.
    
    Args:
        system_message: System instruction
        user_message: User content
        
    Returns:
        Formatted prompt string
    """
    return f"<s>[INST] {system_message}\n\n{user_message} [/INST]"


def load_mistral_model(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
    device: Optional[str] = None,
    torch_dtype: torch.dtype = torch.float16
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Mistral model and tokenizer with standard settings.
    
    Args:
        model_name: Name of the Mistral model
        device: Target device
        torch_dtype: PyTorch data type
        
    Returns:
        Tuple of (model, tokenizer)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading Mistral model {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None
    )
    
    if device != "cuda":  # Only move manually if not using device_map
        model = model.to(device)
    
    return model, tokenizer


def extract_json_from_mistral_response(response_text: str) -> Optional[str]:
    """
    Extract JSON object from Mistral model response.
    Handles various response formats and cleans up the JSON.
    
    Args:
        response_text: Raw response from Mistral model
        
    Returns:
        Extracted JSON string or None if not found
    """
    if not response_text:
        return None
    
    # Clean the response
    text = response_text.strip()
    
    # Method 1: Find JSON block between braces
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        json_text = text[brace_start:brace_end + 1]
        # Validate it's proper JSON
        try:
            json.loads(json_text)
            return json_text
        except json.JSONDecodeError:
            pass
    
    # Method 2: Extract using regex patterns
    # Look for JSON-like structures
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            json.loads(match)
            return match
        except json.JSONDecodeError:
            continue
    
    # Method 3: Try to construct JSON from key-value pairs
    try:
        json_dict = {}
        
        # Common patterns for key-value extraction
        patterns = [
            r'"(\w+)"\s*:\s*"([^"]*)"',  # "key": "value"
            r'"(\w+)"\s*:\s*(\d+\.?\d*)',  # "key": number
            r'"(\w+)"\s*:\s*(true|false)',  # "key": boolean
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for key, value in matches:
                if pattern.endswith('*)"'):  # String values
                    json_dict[key] = value
                elif pattern.endswith('*)')  and ('.' in value or value.isdigit()):  # Numeric values
                    json_dict[key] = float(value) if '.' in value else int(value)
                else:  # Boolean values
                    json_dict[key] = value.lower() == 'true'
        
        if json_dict:
            return json.dumps(json_dict)
    
    except Exception as e:
        logger.debug(f"Failed to extract JSON with regex: {e}")
    
    return None


def validate_mistral_response(
    response_dict: Dict[str, Any],
    required_keys: Optional[List[str]] = None,
    optional_keys: Optional[List[str]] = None
) -> bool:
    """
    Validate Mistral model response structure.
    
    Args:
        response_dict: Parsed JSON response
        required_keys: Keys that must be present
        optional_keys: Keys that are allowed but not required
        
    Returns:
        True if response is valid
    """
    if not isinstance(response_dict, dict):
        return False
    
    # Check required keys
    if required_keys:
        for key in required_keys:
            if key not in response_dict:
                logger.debug(f"Missing required key: {key}")
                return False
    
    # Validate specific field types if present
    if 'confidence' in response_dict:
        try:
            conf = float(response_dict['confidence'])
            if not (0.0 <= conf <= 1.0):
                logger.debug(f"Confidence value out of range: {conf}")
                return False
        except (ValueError, TypeError):
            logger.debug("Invalid confidence value type")
            return False
    
    return True


def generate_mistral_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    device: Optional[str] = None
) -> str:
    """
    Generate response using Mistral model with standard parameters.
    
    Args:
        model: Loaded Mistral model
        tokenizer: Corresponding tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        device: Target device
        
    Returns:
        Generated text (excluding input prompt)
    """
    device = device or next(model.parameters()).device
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decode only the new tokens
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text.strip()


def normalize_conscious_state(state: Dict[str, Any], language: str = "en") -> Dict[str, Any]:
    """
    Normalize conscious state components for consistency.
    
    Args:
        state: Raw state dictionary
        language: Target language for normalization
        
    Returns:
        Normalized state dictionary
    """
    # Default values by language
    defaults = {
        "en": {
            "goal": "understand_input",
            "emotion": "neutral",
            "confidence": 0.5,
            "thought": "Processing input...",
            "memory": []
        },
        "es": {
            "goal": "entender_entrada", 
            "emotion": "neutral",
            "confidence": 0.5,
            "thought": "Procesando entrada...",
            "memory": []
        }
    }
    
    lang_defaults = defaults.get(language, defaults["en"])
    normalized = {}
    
    # Normalize each field
    for key, default_value in lang_defaults.items():
        value = state.get(key, default_value)
        
        if key == "confidence":
            try:
                confidence = float(value)
                normalized[key] = max(0.0, min(1.0, confidence))
            except (ValueError, TypeError):
                normalized[key] = default_value
        elif key == "memory":
            if isinstance(value, list):
                normalized[key] = value
            else:
                normalized[key] = [str(value)] if value else default_value
        else:
            normalized[key] = str(value) if value else default_value
    
    return normalized


def log_mistral_inference(
    input_prompt: str,
    generated_response: str,
    extracted_json: Optional[str],
    validation_result: bool,
    model_name: str = "Mistral"
) -> None:
    """
    Log Mistral inference details for debugging.
    
    Args:
        input_prompt: Input prompt sent to model
        generated_response: Raw model response
        extracted_json: Extracted JSON string
        validation_result: Whether validation passed
        model_name: Name of the model for logging
    """
    logger.debug(f"=== {model_name} Inference Log ===")
    logger.debug(f"Prompt length: {len(input_prompt)} chars")
    logger.debug(f"Response length: {len(generated_response)} chars")
    logger.debug(f"JSON extracted: {'Yes' if extracted_json else 'No'}")
    logger.debug(f"Validation passed: {validation_result}")
    
    if extracted_json:
        logger.debug(f"Extracted JSON: {extracted_json}")
    else:
        logger.debug(f"Raw response: {generated_response[:200]}{'...' if len(generated_response) > 200 else ''}")