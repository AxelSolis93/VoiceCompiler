import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import whisper
import re
import noisereduce as nr
import librosa
import soundfile as sf
import time
from difflib import SequenceMatcher

SAMPLE_RATE = 44100
WAKE_DURATION = 3  # segundos para detectar wake word
COMMAND_DURATION = 8  # segundos para grabar comando
WAKE_WORD = "oye compilador"

class VoiceCompiler:
    def __init__(self):
        self.modelo = whisper.load_model("base")
        self.is_active = True
        
        # Patrones de comandos válidos con variantes
        self.comandos_plantillas = {
            "bucle": {
                "palabras_clave": ["haz", "bucle", "del", "al"],
                "patron": r"(\w+)\s+un\s+(\w+)\s+del\s+(\d+)\s+al\s+(\d+)",
                "estructura": "haz un bucle del {inicio} al {fin}",
                "accion": "bucle"
            },
            "variable": {
                "palabras_clave": ["declara", "variable", "igual"],
                "patron": r"(\w+)\s+una\s+(\w+)\s+(\w+)\s+(\w+)\s+a\s+(\w+)",
                "estructura": "declara una variable {nombre} igual a {valor}",
                "accion": "variable"
            },
            "funcion": {
                "palabras_clave": ["define", "función", "llamada"],
                "patron": r"(\w+)\s+una\s+(\w+)\s+(\w+)\s+(\w+)",
                "estructura": "define una función llamada {nombre}",
                "accion": "funcion"
            },
            "condicional": {
                "palabras_clave": ["si", "mayor", "que"],
                "patron": r"si\s+(\w+)\s+es\s+(\w+)\s+que\s+(\w+)",
                "estructura": "si {var1} es mayor que {var2}",
                "accion": "condicional"
            },
            "mensaje": {
                "palabras_clave": ["muestra", "mensaje"],
                "patron": r"(\w+)\s+el\s+(\w+)\s+(.+)",
                "estructura": "muestra el mensaje {texto}",
                "accion": "mensaje"
            },
            "terminar": {
                "palabras_clave": ["terminar", "salir", "cerrar"],
                "patron": r"(terminar|salir|cerrar)",
                "estructura": "terminar",
                "accion": "terminar"
            }
        }
        
        # Diccionario de correcciones comunes
        self.correcciones = {
            "hasz": "haz", "has": "haz", "az": "haz",
            "bocle": "bucle", "bukle": "bucle", "buble": "bucle",
            "del": "del", "de": "del", "dell": "del",
            "al": "al", "a": "al", "all": "al",
            "declara": "declara", "dekla": "declara", "deklara": "declara",
            "variable": "variable", "variabel": "variable", "bariable": "variable",
            "igual": "igual", "igua": "igual", "ygual": "igual",
            "define": "define", "defin": "define", "defiene": "define",
            "función": "función", "funcion": "función", "funcioon": "función",
            "llamada": "llamada", "yamada": "llamada", "lamada": "llamada",
            "muestra": "muestra", "mostra": "muestra", "muesta": "muestra",
            "mensaje": "mensaje", "mensage": "mensaje", "mesaje": "mensaje",
            "mayor": "mayor", "mallor": "mayor", "maor": "mayor",
            "que": "que", "qe": "que", "ke": "que"
        }
        
    def grabar_audio_corto(self, duracion, nombre_archivo="temp_audio.wav"):
        """Graba audio por tiempo limitado"""
        print(f"🎧 Escuchando por {duracion} segundos...")
        grabacion = sd.rec(int(duracion * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        write(nombre_archivo, SAMPLE_RATE, grabacion)
        return nombre_archivo
    
    def limpiar_audio(self, ruta_original, ruta_limpia="audio_limpio.wav"):
        """Limpia el audio del ruido"""
        try:
            audio, sr = librosa.load(ruta_original, sr=None)
            reducido = nr.reduce_noise(y=audio, sr=sr)
            sf.write(ruta_limpia, reducido, sr)
            return ruta_limpia
        except:
            return ruta_original
    
    def transcribir_audio(self, ruta_audio):
        """Transcribe audio a texto"""
        try:
            resultado = self.modelo.transcribe(ruta_audio, language="es")
            return resultado["text"].lower().strip()
        except:
            return ""
    
    def detectar_wake_word(self, texto):
        """Detecta si el texto contiene la palabra de activación"""
        texto_limpio = re.sub(r'[^\w\s]', '', texto.lower())
        palabras_activacion = ["oye compilador", "oye compiler", "hey compilador"]
        
        for palabra in palabras_activacion:
            if palabra in texto_limpio:
                return True
        return False
    
    def corregir_texto(self, texto):
        """Corrige errores comunes de transcripción"""
        palabras = texto.split()
        palabras_corregidas = []
        
        for palabra in palabras:
            palabra_limpia = re.sub(r'[^\w]', '', palabra.lower())
            if palabra_limpia in self.correcciones:
                palabras_corregidas.append(self.correcciones[palabra_limpia])
            else:
                # Buscar la palabra más similar
                mejor_coincidencia = self.encontrar_palabra_similar(palabra_limpia)
                if mejor_coincidencia:
                    palabras_corregidas.append(mejor_coincidencia)
                else:
                    palabras_corregidas.append(palabra_limpia)
        
        return " ".join(palabras_corregidas)
    
    def encontrar_palabra_similar(self, palabra, umbral=0.6):
        """Encuentra la palabra más similar usando fuzzy matching"""
        mejor_coincidencia = None
        mejor_score = 0
        
        for palabra_correcta in self.correcciones.values():
            score = SequenceMatcher(None, palabra, palabra_correcta).ratio()
            if score > mejor_score and score >= umbral:
                mejor_score = score
                mejor_coincidencia = palabra_correcta
        
        return mejor_coincidencia
    
    def detectar_comando_inteligente(self, texto):
        """Detecta comandos usando fuzzy matching y palabras clave"""
        texto_corregido = self.corregir_texto(texto)
        print(f"🔧 Texto corregido: '{texto_corregido}'")
        
        mejor_comando = None
        mejor_score = 0
        
        for nombre_comando, plantilla in self.comandos_plantillas.items():
            # Contar cuántas palabras clave están presentes
            palabras_encontradas = 0
            for palabra_clave in plantilla["palabras_clave"]:
                if palabra_clave in texto_corregido:
                    palabras_encontradas += 1
            
            # Calcular score basado en palabras clave encontradas
            score = palabras_encontradas / len(plantilla["palabras_clave"])
            
            if score > mejor_score and score >= 0.5:  # Al menos 50% de coincidencia
                mejor_score = score
                mejor_comando = nombre_comando
        
        return mejor_comando, texto_corregido
    
    def extraer_parametros(self, texto_corregido, tipo_comando):
        """Extrae parámetros del comando corregido"""
        if tipo_comando == "bucle":
            # Buscar números en el texto
            numeros = re.findall(r'\d+', texto_corregido)
            if len(numeros) >= 2:
                return {"inicio": numeros[0], "fin": numeros[1]}
        
        elif tipo_comando == "variable":
            # Buscar patrón variable nombre = valor
            palabras = texto_corregido.split()
            if "variable" in palabras:
                idx = palabras.index("variable")
                if idx + 1 < len(palabras):
                    nombre = palabras[idx + 1]
                    # Buscar valor después de "igual"
                    if "igual" in palabras:
                        idx_igual = palabras.index("igual")
                        if idx_igual + 1 < len(palabras):
                            valor = palabras[idx_igual + 1]
                            return {"nombre": nombre, "valor": valor}
        
        elif tipo_comando == "funcion":
            # Buscar nombre después de "llamada"
            palabras = texto_corregido.split()
            if "llamada" in palabras:
                idx = palabras.index("llamada")
                if idx + 1 < len(palabras):
                    return {"nombre": palabras[idx + 1]}
        
        elif tipo_comando == "condicional":
            # Buscar variables en la condición
            palabras = texto_corregido.split()
            if "si" in palabras and "mayor" in palabras:
                idx_si = palabras.index("si")
                idx_mayor = palabras.index("mayor")
                if idx_si + 1 < len(palabras) and idx_mayor + 2 < len(palabras):
                    var1 = palabras[idx_si + 1]
                    var2 = palabras[idx_mayor + 2]  # después de "que"
                    return {"var1": var1, "var2": var2}
        
        elif tipo_comando == "mensaje":
            # Extraer todo después de "mensaje"
            if "mensaje" in texto_corregido:
                partes = texto_corregido.split("mensaje", 1)
                if len(partes) > 1:
                    mensaje = partes[1].strip()
                    return {"texto": mensaje}
        
        return None
    
    def abrir_archivo(self, nombre_archivo):
        """Abre el archivo de código generado"""
        import os
        import platform
        
        try:
            sistema = platform.system()
            if sistema == "Windows":
                os.startfile(nombre_archivo)
            elif sistema == "Darwin":  # macOS
                os.system(f"open {nombre_archivo}")
            else:  # Linux
                os.system(f"xdg-open {nombre_archivo}")
            print(f"📂 Archivo abierto: {nombre_archivo}")
        except Exception as e:
            print(f"❌ No pude abrir el archivo: {e}")
            print(f"💡 Puedes abrir manualmente: {nombre_archivo}")

    def interpretar_comando_inteligente(self, tipo_comando, parametros):
        """Interpreta comandos usando el sistema inteligente"""
        if tipo_comando == "bucle" and parametros:
            inicio = parametros["inicio"]
            fin = int(parametros["fin"]) + 1
            return f"for i in range({inicio}, {fin}):\n    print(i)\n"
        
        elif tipo_comando == "variable" and parametros:
            nombre = parametros["nombre"]
            valor = parametros["valor"]
            return f"{nombre} = {valor}\n"
        
        elif tipo_comando == "funcion" and parametros:
            nombre = parametros["nombre"]
            return f"def {nombre}():\n    pass\n"
        
        elif tipo_comando == "condicional" and parametros:
            var1 = parametros["var1"]
            var2 = parametros["var2"]
            return f"if {var1} > {var2}:\n    print('{var1} es mayor')\n"
        
        elif tipo_comando == "mensaje" and parametros:
            texto = parametros["texto"]
            return f"print(\"{texto}\")\n"
        
        elif tipo_comando == "terminar":
            print("👋 ¡Hasta luego!")
            self.is_active = False
            return None
        
        return None
    
    def guardar_codigo(self, codigo, nombre_archivo="codigo_generado.py"):
        """Guarda el código generado"""
        with open(nombre_archivo, "a", encoding="utf-8") as archivo:
            archivo.write(codigo)
        print(f"✅ Código guardado en '{nombre_archivo}'")
    
    def esperar_activacion(self):
        """Bucle principal que espera la palabra de activación"""
        print("🤖 Compilador de Voz Iniciado")
        print("💡 Di 'Oye compilador' para empezar")
        print("💡 Comandos disponibles:")
        print("   - Haz un bucle del X al Y")
        print("   - Declara una variable X igual a Y")
        print("   - Define una función llamada X")
        print("   - Si X es mayor que Y")
        print("   - Muestra el mensaje 'texto'")
        print("   - Terminar/Salir/Cerrar")
        print("-" * 50)
        
        while self.is_active:
            # Escucha corta para detectar wake word
            ruta_wake = self.grabar_audio_corto(WAKE_DURATION, "wake_check.wav")
            ruta_wake_limpia = self.limpiar_audio(ruta_wake, "wake_limpia.wav")
            texto_wake = self.transcribir_audio(ruta_wake_limpia)
            
            if texto_wake:
                print(f"🎤 Escuché: '{texto_wake}'")
                
                if self.detectar_wake_word(texto_wake):
                    print("🔥 ¡Palabra de activación detectada!")
                    self.procesar_comando()
                else:
                    print("💭 Esperando 'Oye compilador'...")
            
            time.sleep(0.5)
    
    def procesar_comando(self):
        """Procesa el comando después de detectar wake word"""
        print(f"🔴 Grabando comando por {COMMAND_DURATION} segundos...")
        print("💬 Dime tu comando ahora (puedes hablar libremente):")
        
        # Graba el comando
        ruta_comando = self.grabar_audio_corto(COMMAND_DURATION, "comando.wav")
        ruta_comando_limpia = self.limpiar_audio(ruta_comando, "comando_limpio.wav")
        texto_completo = self.transcribir_audio(ruta_comando_limpia)
        
        if texto_completo:
            print(f"📝 Texto original: '{texto_completo}'")
            
            # Detectar comando usando sistema inteligente
            tipo_comando, texto_corregido = self.detectar_comando_inteligente(texto_completo)
            
            if tipo_comando:
                print(f"🎯 Comando detectado: {tipo_comando}")
                
                # Extraer parámetros
                parametros = self.extraer_parametros(texto_corregido, tipo_comando)
                
                if parametros or tipo_comando == "terminar":
                    print(f"📋 Parámetros: {parametros}")
                    
                    # Generar código
                    codigo = self.interpretar_comando_inteligente(tipo_comando, parametros)
                    
                    if codigo:
                        print(f"🐍 Código generado:\n{codigo}")
                        self.guardar_codigo(codigo)
                        self.abrir_archivo("codigo_generado.py")
                    elif tipo_comando == "terminar":
                        pass  # Ya se maneja en interpretar_comando_inteligente
                    else:
                        print("❌ Error al generar código")
                else:
                    print("❌ No pude extraer los parámetros necesarios")
                    print("💡 Intenta ser más específico con los valores")
            else:
                print("❌ No detecté ningún comando válido")
                print("💡 Comandos disponibles:")
                print("   - Haz un bucle del X al Y")
                print("   - Declara una variable X igual a Y")
                print("   - Define una función llamada X")
                print("   - Si X es mayor que Y")
                print("   - Muestra el mensaje 'texto'")
                print("   - Terminar")
            
            print("✨ Listo para el siguiente comando")
            print("-" * 50)
        else:
            print("❌ No pude entender el comando")

def main():
    compilador = VoiceCompiler()
    try:
        compilador.esperar_activacion()
    except KeyboardInterrupt:
        print("\n👋 Saliendo del compilador...")

if __name__ == "__main__":
    main()