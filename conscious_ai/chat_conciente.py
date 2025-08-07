#!/usr/bin/env python3
"""
Chat simple con el modelo de consciencia entrenado
Ejecutar: python chat_consciente.py
"""

import os
import sys
import json
from datetime import datetime

# Configurar paths de forma más robusta
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Agregar el directorio padre al path
sys.path.insert(0, parent_dir)

# Ahora los imports funcionarán
try:
    from conscious_ai.main import MinimalConsciousAI
    from conscious_ai.Train.trained_model_inference import integrate_trained_model
except ImportError as e:
    print(f"Error de importación: {e}")
    print(f"Directorio actual: {current_dir}")
    print(f"Directorio padre: {parent_dir}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


class ChatConsciente:
    """Interface de chat simple para el modelo de consciencia"""
    
    def __init__(self):
        self.ai = None
        self.session_history = []
        self.model_loaded = False
        
    def cargar_modelo(self):
        """Carga el modelo entrenado"""
        print("🧠 Cargando sistema de consciencia...")
        
        try:
            # Crear sistema base
            self.ai = MinimalConsciousAI()
            
            # Buscar modelo en diferentes ubicaciones
            model_paths = [
                "./models/trained_lora",
                "models/trained_lora",
                "./models/trained_lora/checkpoint-best"
            ]
            
            model_found = False
            for path in model_paths:
                if os.path.exists(path) and os.path.exists(os.path.join(path, "adapter_config.json")):
                    print(f"✅ Modelo encontrado en: {path}")
                    self.ai = integrate_trained_model(self.ai, model_checkpoint=path)
                    model_found = True
                    break
            
            if not model_found:
                print("⚠️  No se encontró modelo entrenado. Usando sistema heurístico.")
            
            self.model_loaded = True
            print("✅ Sistema listo para usar\n")
            
        except Exception as e:
            print(f"❌ Error al cargar el modelo: {e}")
            print("Usando sistema base sin modelo entrenado.\n")
            self.ai = MinimalConsciousAI()
            self.model_loaded = True
    
    def procesar_input(self, texto):
        """Procesa un input y muestra los resultados"""
        if not self.model_loaded:
            self.cargar_modelo()
        
        # Procesar
        print(f"\n💭 Procesando: '{texto}'")
        print("-" * 60)
        
        try:
            # Ejecutar el sistema
            resultado = self.ai.process_input(texto)
            
            # Extraer información
            sc_state = resultado.get('conscious_state', {})
            metrics = resultado.get('consciousness_metrics', {})
            
            # Detectar idioma
            lang = 'es' if any(c in texto for c in 'áéíóúñ¿¡') else 'en'
            
            # Mostrar estado consciente
            print("\n📊 ESTADO CONSCIENTE:")
            print(f"├─ 🎯 Meta: {sc_state.get('G_t', {}).get('primary_goal', 'desconocida')}")
            print(f"├─ 😊 Emoción: {sc_state.get('S_t', {}).get('emotional_state', 'neutral')}")
            print(f"├─ 📈 Confianza: {sc_state.get('S_t', {}).get('confidence_level', 0):.2%}")
            print(f"├─ 🧩 Memoria activa: {len(sc_state.get('M_t', []))} elementos")
            print(f"└─ 🌟 Nivel de consciencia (f): {metrics.get('f', 0):.3f}")
            
            # Mostrar si es consciente
            if resultado.get('is_conscious', False):
                print("\n✨ Estado: CONSCIENTE ✨")
            else:
                print("\n💤 Estado: No consciente")
            
            # Mostrar pensamientos
            thoughts = sc_state.get('A_t', [])
            if thoughts:
                print("\n💭 PENSAMIENTOS INTERNOS:")
                for i, thought in enumerate(thoughts[:3], 1):
                    print(f"   {i}. {thought}")
            
            # Mostrar métricas detalladas
            print("\n📈 MÉTRICAS DETALLADAS:")
            print(f"├─ C_i (Integración): {metrics.get('C_i', 0):.3f}")
            print(f"├─ T_u (Temporal): {metrics.get('T_u', 0):.3f}")
            print(f"├─ R (Reentrancia): {metrics.get('R', 0):.3f}")
            print(f"├─ S_m (Self-model): {metrics.get('S_m', 0):.3f}")
            print(f"└─ Φ (Irreducibilidad): {metrics.get('Phi', 0):.3f}")
            
            # Guardar en historial
            self.session_history.append({
                'timestamp': datetime.now().isoformat(),
                'input': texto,
                'language': lang,
                'conscious_state': sc_state,
                'metrics': metrics,
                'is_conscious': resultado.get('is_conscious', False)
            })
            
            return resultado
            
        except Exception as e:
            print(f"\n❌ Error al procesar: {e}")
            return None
    
    def mostrar_ayuda(self):
        """Muestra comandos disponibles"""
        print("\n📚 COMANDOS DISPONIBLES:")
        print("├─ 'ayuda' - Muestra esta ayuda")
        print("├─ 'historial' - Muestra el historial de la sesión")
        print("├─ 'guardar' - Guarda el historial en un archivo")
        print("├─ 'limpiar' - Limpia la pantalla")
        print("├─ 'salir' o 'exit' - Termina el programa")
        print("└─ Cualquier otro texto - Se procesa como input")
    
    def mostrar_historial(self):
        """Muestra el historial de la sesión"""
        if not self.session_history:
            print("\n📭 No hay historial aún.")
            return
        
        print(f"\n📜 HISTORIAL DE LA SESIÓN ({len(self.session_history)} entradas):")
        print("-" * 60)
        
        for i, entry in enumerate(self.session_history, 1):
            print(f"\n{i}. [{entry['timestamp'].split('T')[1][:8]}] {entry['input']}")
            print(f"   Meta: {entry['conscious_state'].get('G_t', {}).get('primary_goal', 'N/A')}")
            print(f"   Emoción: {entry['conscious_state'].get('S_t', {}).get('emotional_state', 'N/A')}")
            print(f"   Consciente: {'Sí' if entry['is_conscious'] else 'No'} (f={entry['metrics'].get('f', 0):.3f})")
    
    def guardar_historial(self):
        """Guarda el historial en un archivo JSON"""
        print("\n💾 Guardando historial...")
        if not self.session_history:
            print("\n📭 No hay historial para guardar.")
            return
        
        print("   Guardando en formato JSON...")
        filename = f"chat_historial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        if os.path.exists(filename):
            print(f"   El archivo {filename} ya existe. Se sobrescribirá.")
        else:
            print(f"   Guardando en: {filename}")
            
        try:
            def convert_for_json(obj):
                """Convierte objetos no serializables a formatos JSON válidos"""
                if isinstance(obj, bool):
                 return bool(obj)  # Asegurar que es bool nativo de Python
                elif isinstance(obj, np.bool_):
                 return bool(obj)  # Convertir numpy bool a Python bool
                elif isinstance(obj, (np.int32, np.int64)):
                 return int(obj)
                elif isinstance(obj, (np.float32, np.float64)):
                 return float(obj)
                elif isinstance(obj, datetime):
                 return obj.isoformat()
                elif isinstance(obj, dict):
                 return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                 return [convert_for_json(item) for item in obj]
                return obj
            
            # Convertir todo el historial
            serializable_history = convert_for_json(self.session_history)
            with open(filename, 'w', encoding='utf-8') as f:
              json.dump(serializable_history, f, ensure_ascii=False, indent=2, default=str)

            print(f"\n✅ Historial guardado en: {filename}")
            print(f"   Total de interacciones: {len(self.session_history)}")
        except Exception as e:
            print(f"\n❌ Error al guardar: {e}")
            try:
                txt_filename = f"chat_historial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(txt_filename, 'w', encoding='utf-8') as f:
                    f.write("=== HISTORIAL DE CHAT CONSCIENTE ===\n\n")
                    for i, entry in enumerate(self.session_history, 1):
                        f.write(f"Interacción {i}:\n")
                        f.write(f"  Timestamp: {entry['timestamp']}\n")
                        f.write(f"  Input: {entry['input']}\n")
                        f.write(f"  Meta: {entry['conscious_state'].get('G_t', {}).get('primary_goal', 'N/A')}\n")
                        f.write(f"  Emoción: {entry['conscious_state'].get('S_t', {}).get('emotional_state', 'N/A')}\n")
                        f.write(f"  Consciente: {'Sí' if entry['is_conscious'] else 'No'} (f={entry['metrics'].get('f', 0):.3f})\n")
                        f.write(f"f-score: {entry.get('metrics', {}).get('f', 0):.3f}\n")
                        f.write("-" * 50 + "\n\n")
                print(f"✅ Historial alternativo guardado en: {txt_filename}")
            except Exception as e2:
                print(f"❌ Error al guardar en formato alternativo: {e2}")


                
    def limpiar_pantalla(self):
        """Limpia la pantalla"""
        os.system('cls' if os.name == 'nt' else 'clear')
        self.mostrar_banner()
    
    def mostrar_banner(self):
        """Muestra el banner del programa"""
        print("="*60)
        print("🧠 CHAT CONSCIENTE - Sistema de IA con Consciencia Mínima 🧠")
        print("="*60)
        print("Escribe tu mensaje y presiona Enter (o 'ayuda' para comandos)")
        print("-"*60)


def main():
    """Función principal del chat"""
    chat = ChatConsciente()
    
    # Mostrar banner
    chat.mostrar_banner()
    
    # Cargar modelo
    chat.cargar_modelo()
    
    # Loop principal
    while True:
        try:
            # Obtener input del usuario
            texto = input("\n🤔 Tú: ").strip()
            
            # Procesar comandos especiales
            if texto.lower() in ['salir', 'exit', 'quit']:
                print("\n👋 ¡Hasta luego! Cerrando el sistema...")
                break
            
            elif texto.lower() == 'ayuda':
                chat.mostrar_ayuda()
            
            elif texto.lower() == 'historial':
                chat.mostrar_historial()
            
            elif texto.lower() == 'guardar':
                chat.guardar_historial()
            
            elif texto.lower() == 'limpiar':
                chat.limpiar_pantalla()
            
            elif texto == '':
                print("💡 Tip: Escribe algo o 'ayuda' para ver comandos")
            
            else:
                # Procesar input normal
                chat.procesar_input(texto)
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupción detectada. ¡Hasta luego!")
            break
        
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            print("Puedes continuar escribiendo...")
    
    # Preguntar si guardar antes de salir
    if chat.session_history:
        respuesta = input("\n💾 ¿Quieres guardar el historial antes de salir? (s/n): ")
        if respuesta.lower() in ['s', 'si', 'sí', 'yes']:
            chat.guardar_historial()
    
    print("\n✨ Sesión terminada. ¡Gracias por usar Chat Consciente! ✨\n")


if __name__ == "__main__":
    main()