# 🦙 LLaMA 3.2 1B – Fine-Tuning en Español (QLoRA + Unsloth)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

Fine-tuning eficiente del modelo **LLaMA 3.2 1B Instruct** en español usando **QLoRA (4-bit)** con la librería **Unsloth**, ejecutado en el cluster **HPC-UCR**.

---

## 📋 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Inicio Rápido](#-inicio-rápido)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Formato de Datos](#-formato-de-datos)
- [Entrenamiento](#-entrenamiento)
- [Inferencia](#-inferencia)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Resultados](#-resultados)
- [Resolución de Problemas](#-resolución-de-problemas)
- [Contribuciones](#-contribuciones)
- [Autor](#-autor)

---

## 🧠 Descripción General

Este proyecto adapta el modelo **LLaMA 3.2 1B Instruct** al **español** utilizando fine-tuning eficiente con **QLoRA (cuantización de 4 bits)** a través de la librería **Unsloth**, ejecutado en el cluster **HPC-UCR**.

### Características Principales

✅ **Eficiencia de Memoria**: Cuantización de 4 bits reduce el uso de memoria en ~75%
✅ **Entrenamiento Rápido**: Optimizaciones de Unsloth aceleran el entrenamiento
✅ **Fácil Configuración**: Scripts automatizados para setup
✅ **Flexible**: Configuración centralizada y fácil de modificar
✅ **Reproducible**: Semillas aleatorias y configuración documentada
✅ **HPC-Ready**: Script SLURM incluido para clusters

### Especificaciones Técnicas

| Componente | Descripción |
|-----------|-------------|
| **Framework** | PyTorch 2.7.1, TorchVision 0.22.1, TorchAudio 2.7.1 |
| **Transformers** | HuggingFace Transformers |
| **Estrategia** | QLoRA (4-bit) + Low Rank Adapters |
| **Modelo Base** | `meta-llama/Llama-3.2-1B-Instruct` |
| **Longitud de Secuencia** | 4096 tokens |
| **Optimizador** | AdamW con warmup + cosine decay |
| **Formato de Datos** | JSONL (instruction, input, output) |
| **Infraestructura** | HPC-UCR GPU partition (A100 80GB) |
| **Evaluación** | Eval Loss & Perplexity |

---

## 🚀 Inicio Rápido

```bash
# 1. Clonar el repositorio
git clone <repository-url>
cd llama32_qlora

# 2. Ejecutar setup automático
bash setup.sh

# 3. Colocar tus datos de entrenamiento
# Coloca tu dataset en: data/base.jsonl

# 4. Entrenar (local)
python scripts/train_llama32_gpu.py

# O entrenar en HPC con SLURM
sbatch scripts/train_block_full_gpu.sbatch

# 5. Ejecutar inferencia
python scripts/infer_llama.py \
    --model_path outputs/llama32_qlora \
    --prompt "¿Qué es Python?"
```

---

## 📦 Instalación

### Opción 1: Setup Automático (Recomendado)

```bash
bash setup.sh
```

El script automatizado:
- ✅ Verifica dependencias del sistema
- ✅ Crea entorno virtual
- ✅ Instala todas las dependencias
- ✅ Configura estructura de directorios
- ✅ Verifica instalación de GPU/CUDA

### Opción 2: Instalación Manual

Ver la [Guía de Instalación Detallada](INSTALL.md) para instrucciones paso a paso.

#### Requisitos Mínimos

- Python 3.8+
- NVIDIA GPU con 12GB+ VRAM
- CUDA 11.8+
- 20GB de espacio en disco

#### Instalación Rápida

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

Ver [INSTALL.md](INSTALL.md) para detalles completos y troubleshooting.

---

## ⚙️ Configuración

### Archivo de Configuración

Todas las configuraciones están centralizadas en `config.py`:

```python
# Editar config.py para personalizar

# Modelo y datos
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_PATH = "data/base.jsonl"

# Parámetros de entrenamiento
NUM_TRAIN_EPOCHS = 60
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 2

# Parámetros LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
```

### Variables de Entorno

También puedes usar variables de entorno para override:

```bash
export MODEL="meta-llama/Llama-3.2-1B-Instruct"
export DATA="./data/mi_dataset.jsonl"
export EPOCHS=40
export OUT="./outputs/mi_modelo"

python scripts/train_llama32_gpu.py
```

### Verificar Configuración

```bash
# Ver configuración actual
python config.py
```

---

## 📊 Formato de Datos

### Formato JSONL Requerido

Tu dataset debe estar en formato JSONL (JSON Lines) con los siguientes campos:

```jsonl
{"instruction": "Traduce al inglés", "input": "Hola mundo", "output": "Hello world"}
{"instruction": "¿Qué es Python?", "output": "Python es un lenguaje de programación..."}
{"instruction": "Resume este texto", "input": "Texto largo...", "output": "Resumen..."}
```

### Campos

- **`instruction`** (requerido): La instrucción o pregunta
- **`input`** (opcional): Contexto o entrada adicional
- **`output`** (requerido): La respuesta esperada

### Ejemplo de Dataset

Crea `data/base.jsonl`:

```jsonl
{"instruction": "¿Qué es machine learning?", "output": "Machine learning es una rama de la inteligencia artificial que permite a las computadoras aprender de datos sin ser programadas explícitamente."}
{"instruction": "Traduce al inglés", "input": "Buenos días", "output": "Good morning"}
{"instruction": "Resume en una frase", "input": "Python es un lenguaje de programación de alto nivel, interpretado y de propósito general.", "output": "Python es un lenguaje versátil de alto nivel."}
```

### Validar Dataset

```bash
# Verificar formato del dataset
python - << 'EOF'
import json

with open('data/base.jsonl', 'r') as f:
    for i, line in enumerate(f, 1):
        try:
            data = json.loads(line)
            assert 'instruction' in data, f"Línea {i}: falta 'instruction'"
            assert 'output' in data, f"Línea {i}: falta 'output'"
            print(f"✅ Línea {i}: OK")
        except Exception as e:
            print(f"❌ Línea {i}: {e}")
EOF
```

---

## 🎯 Entrenamiento

### Entrenamiento Local

```bash
# Activar entorno virtual
source venv/bin/activate

# Entrenar con configuración por defecto
python scripts/train_llama32_gpu.py

# O con parámetros personalizados
EPOCHS=40 LEARNING_RATE=1e-4 python scripts/train_llama32_gpu.py
```

### Entrenamiento en HPC con SLURM

#### 1. Configurar Usuario

Edita `scripts/train_block_full_gpu.sbatch`:

```bash
# Cambiar estas líneas según tu sistema
USER_HOME="${HOME}"
PROJECT_ROOT="${USER_HOME}/llama32_qlora"

# Configurar email (opcional)
# #SBATCH --mail-user=tu.email@ejemplo.com
# #SBATCH --mail-type=END,FAIL
```

#### 2. Enviar Job

```bash
cd scripts
sbatch train_block_full_gpu.sbatch
```

#### 3. Monitorear Job

```bash
# Ver estado del job
squeue -u $USER

# Ver output en tiempo real
tail -f llama32_qlora_full_*.out

# Ver logs completos
less llama32_qlora_full_*.out
```

### Parámetros de Entrenamiento

| Parámetro | Valor por Defecto | Descripción |
|-----------|-------------------|-------------|
| `EPOCHS` | 60 | Número de épocas de entrenamiento |
| `MAX_STEPS` | 0 | Pasos máximos (0 = usar épocas completas) |
| `LEARNING_RATE` | 2e-4 | Tasa de aprendizaje |
| `BATCH_SIZE` | 2 | Tamaño de batch por dispositivo |
| `GRAD_ACCUM` | 4 | Pasos de acumulación de gradientes |
| `EVAL_STEPS` | 200 | Frecuencia de evaluación |
| `LORA_R` | 16 | Rango de LoRA |
| `LORA_ALPHA` | 32 | Alpha de LoRA |

---

## 🔮 Inferencia

### Inferencia Básica

```bash
python scripts/infer_llama.py \
    --model_path outputs/llama32_qlora \
    --prompt "¿Qué es Python?"
```

### Modo Interactivo

```bash
python scripts/infer_llama.py \
    --model_path outputs/llama32_qlora \
    --interactive
```

### Parámetros Personalizados

```bash
python scripts/infer_llama.py \
    --model_path outputs/llama32_qlora \
    --prompt "Explica qué es machine learning" \
    --max_tokens 300 \
    --temperature 0.8 \
    --top_p 0.95
```

### Opciones de Inferencia

| Parámetro | Por Defecto | Descripción |
|-----------|-------------|-------------|
| `--model_path` | (requerido) | Ruta al modelo entrenado |
| `--prompt` | None | Texto de entrada |
| `--interactive` | False | Modo interactivo |
| `--max_tokens` | 200 | Máximo de tokens a generar |
| `--temperature` | 0.7 | Temperatura de sampling (0=determinista) |
| `--top_p` | 0.9 | Nucleus sampling |
| `--top_k` | 50 | Top-k sampling |
| `--repetition_penalty` | 1.1 | Penalización por repetición |
| `--no_cuda` | False | Forzar uso de CPU |

---

## 🗂️ Estructura del Proyecto

```
llama32_qlora/
├── README.md                    # Este archivo
├── INSTALL.md                   # Guía de instalación detallada
├── requirements.txt             # Dependencias de Python
├── config.py                    # Configuración centralizada
├── setup.sh                     # Script de setup automático
├── .gitignore                   # Archivos ignorados por git
│
├── scripts/                     # Scripts de entrenamiento e inferencia
│   ├── train_llama32_gpu.py         # Script principal de entrenamiento
│   ├── train_block_full_gpu.sbatch  # Job SLURM para HPC
│   └── infer_llama.py               # Script de inferencia
│
├── data/                        # Datos de entrenamiento
│   └── base.jsonl                   # Tu dataset (no incluido)
│
├── outputs/                     # Modelos entrenados
│   └── llama32_qlora/               # Checkpoints y adapters
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_summary.json
│
├── logs/                        # Logs de entrenamiento
│   └── *.out / *.err                # Logs de SLURM
│
├── models/                      # Cache de HuggingFace
│   ├── hf_home/
│   └── hf_cache/
│
└── venv/                        # Entorno virtual (creado por setup)
```

---

## 📈 Resultados

El entrenamiento fue ejecutado por **60 épocas** usando QLoRA (4-bit) en una GPU NVIDIA **A100 80GB** (HPC-UCR), con un scheduler de warmup + cosine decay.

### Métricas Finales

| Métrica | Valor Inicial | Valor Final | Mejora |
|---------|---------------|-------------|--------|
| **Eval Loss** | 3.08 | 1.70 | ↓ 45% |
| **Perplexity** | 21.74 | 5.47 | ↓ 75% |
| **Train Loss** | 3.22 | 0.14 | ↓ 96% |

### Progreso por Época

| Época | Train Loss | Eval Loss | Perplexity | Learning Rate |
|------:|-----------:|----------:|-----------:|--------------:|
| 1     | 3.22      | 3.08      | 21.74      | 2.66×10⁻⁵    |
| 10    | 0.21      | 3.01      | 20.37      | 1.71×10⁻⁴    |
| 20    | 0.18      | 2.98      | 19.68      | 1.37×10⁻⁴    |
| 30    | 0.15      | 2.81      | 16.65      | 1.03×10⁻⁴    |
| 40    | 0.15      | 2.67      | 14.38      | 6.88×10⁻⁵    |
| 50    | 0.15      | 2.46      | 11.67      | 3.40×10⁻⁵    |
| 60    | 0.14      | 1.70      | 5.47       | 4.35×10⁻⁸    |

### Observaciones

- ✅ **Reducción consistente** de loss y perplejidad en validación
- ✅ **Sin overfitting**: Mejora continua en set de evaluación
- ✅ **Estabilidad**: Gradient norm estable (4.09 → 0.29)
- ✅ **Convergencia**: Learning rate decae suavemente
- ⚡ **Tiempo de entrenamiento**: ~18 horas (3 bloques × 6 horas en A100)

---

## 🛠️ Resolución de Problemas

### GPU No Detectada

```bash
# Verificar GPU
nvidia-smi

# Verificar CUDA en PyTorch
python -c "import torch; print(torch.cuda.is_available())"

# Si devuelve False, reinstalar PyTorch
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
```

### Out of Memory (OOM)

Edita `config.py`:

```python
# Reducir batch size
PER_DEVICE_TRAIN_BATCH_SIZE = 1

# Aumentar gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 8

# Reducir longitud de secuencia
MAX_SEQ_LENGTH = 2048
```

### Dataset No Encontrado

```bash
# Verificar que existe
ls -lh data/base.jsonl

# Verificar formato
head -n 3 data/base.jsonl

# Validar JSON
python -m json.tool < data/base.jsonl > /dev/null && echo "✅ Valid JSON"
```

### Error al Instalar Unsloth

```bash
# Probar instalación desde source
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# O instalar componentes individualmente
pip install bitsandbytes peft
```

### Ver Más Troubleshooting

Consulta [INSTALL.md](INSTALL.md#troubleshooting) para soluciones detalladas.

---

## 🤝 Contribuciones

Este es un proyecto académico de la Universidad de Costa Rica. Si encuentras problemas o tienes sugerencias:

1. Documenta el problema claramente
2. Incluye pasos para reproducir
3. Adjunta logs relevantes
4. Especifica tu entorno (OS, Python, CUDA, GPU)

---

## 📝 Licencia

Este proyecto es de uso académico. El dataset no está incluido por razones de privacidad y licencia.

---

## 👩‍💻 Autor

**Alison Lobo Salas**
Universidad de Costa Rica (UCR)
📍 San José, Costa Rica

---

## 🙏 Agradecimientos

- **HPC-UCR**: Por proveer la infraestructura computacional
- **Unsloth**: Por la librería de entrenamiento eficiente
- **HuggingFace**: Por Transformers y el ecosistema de modelos
- **Meta**: Por el modelo LLaMA 3.2

---

## 📚 Referencias

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [LLaMA 3.2 Model Card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)

---

## 📋 Changelog

### v2.0.0 (2025-12-04)
- ✨ Configuración centralizada con `config.py`
- ✨ Script de setup automático
- ✨ Documentación mejorada y más clara
- ✨ Mejor manejo de errores
- ✨ Rutas configurables (no hardcodeadas)
- ✨ Modo interactivo de inferencia
- ✨ Guía de instalación detallada
- 🐛 Correcciones de paths y compatibilidad
- 📚 README reorganizado y más accesible

### v1.0.0
- 🎉 Versión inicial del proyecto
- ✅ Entrenamiento funcional con QLoRA
- ✅ Scripts de inferencia básicos

---

**¿Preguntas?** Consulta [INSTALL.md](INSTALL.md) para más detalles o revisa los comentarios en el código.
