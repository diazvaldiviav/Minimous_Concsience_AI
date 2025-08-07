import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from peft import PeftModel

# Cargar modelo
print("Cargando modelo...")
base_model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
model = PeftModel.from_pretrained(base_model, "./models/trained_lora")
model.eval()

# Probar diferentes prompts
test_cases = [
    "Hola como estas",
    '{"input": "Hola como estas"}',
    "Generate JSON for: Hola como estas",
    "Input: 'Hola como estas'\nOutput:",
    "Translate to JSON: Hola como estas"
]

print("\n=== PROBANDO DIFERENTES PROMPTS ===\n")

for prompt in test_cases:
    print(f"Prompt: {prompt}")
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=128, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            do_sample=False,
            temperature=0.7
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Output: {generated}")
    print("-" * 50)