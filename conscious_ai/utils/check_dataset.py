# check_dataset.py
import json

print("=== VERIFICANDO DATASET DE ENTRENAMIENTO ===\n")

with open("training_data.jsonl", 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 3:  # Ver primeros 3 ejemplos
            data = json.loads(line)
            print(f"Ejemplo {i+1}:")
            print(f"Input: {data['input']}")
            print(f"¿Qué debería generar el modelo?")
            print(json.dumps({k:v for k,v in data.items() if k != 'input'}, ensure_ascii=False))
            print("-" * 50)