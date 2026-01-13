"""
Backend Flask para presentación del proyecto LLM-HPC
Incluye autenticación, seguridad y API REST
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import os
import secrets
from functools import wraps
import re

# Importar módulos de seguridad
from security import SecurityHeaders, InputValidator, CSRFProtection
from models import Database

# Inicializar aplicación
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['JWT_EXPIRATION_HOURS'] = 24

# Configurar CORS de manera segura
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5000", "http://127.0.0.1:5000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-CSRF-Token"],
        "supports_credentials": True
    }
})

# Inicializar rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Inicializar componentes de seguridad
security_headers = SecurityHeaders()
input_validator = InputValidator()
csrf_protection = CSRFProtection()
db = Database()

# Middleware de seguridad
@app.after_request
def apply_security_headers(response):
    """Aplicar headers de seguridad a todas las respuestas"""
    return security_headers.add_headers(response)

# Decorador para requerir autenticación
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')

        if not token:
            return jsonify({'message': 'Token faltante'}), 401

        try:
            # Remover 'Bearer ' si está presente
            if token.startswith('Bearer '):
                token = token[7:]

            # Decodificar token
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = db.get_user_by_id(data['user_id'])

            if not current_user:
                return jsonify({'message': 'Usuario no válido'}), 401

        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token expirado'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'message': 'Token inválido'}), 401

        return f(current_user, *args, **kwargs)

    return decorated

# ==================== RUTAS DE AUTENTICACIÓN ====================

@app.route('/api/register', methods=['POST'])
@limiter.limit("5 per hour")
def register():
    """Registro de nuevos usuarios"""
    try:
        data = request.get_json()

        # Validar datos de entrada
        if not data or not data.get('username') or not data.get('password') or not data.get('email'):
            return jsonify({'message': 'Datos incompletos'}), 400

        username = data['username']
        password = data['password']
        email = data['email']

        # Validar formato
        if not input_validator.validate_username(username):
            return jsonify({'message': 'Nombre de usuario inválido (3-20 caracteres alfanuméricos)'}), 400

        if not input_validator.validate_email(email):
            return jsonify({'message': 'Email inválido'}), 400

        if not input_validator.validate_password(password):
            return jsonify({'message': 'Contraseña debe tener al menos 8 caracteres, incluir mayúsculas, minúsculas y números'}), 400

        # Sanitizar entrada
        username = input_validator.sanitize_input(username)
        email = input_validator.sanitize_input(email)

        # Verificar si usuario ya existe
        if db.get_user_by_username(username):
            return jsonify({'message': 'Usuario ya existe'}), 409

        if db.get_user_by_email(email):
            return jsonify({'message': 'Email ya registrado'}), 409

        # Hash de contraseña
        password_hash = generate_password_hash(password, method='pbkdf2:sha256')

        # Crear usuario
        user_id = db.create_user(username, password_hash, email)

        return jsonify({
            'message': 'Usuario registrado exitosamente',
            'user_id': user_id
        }), 201

    except Exception as e:
        return jsonify({'message': f'Error en registro: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    """Autenticación de usuarios"""
    try:
        data = request.get_json()

        if not data or not data.get('username') or not data.get('password'):
            return jsonify({'message': 'Credenciales incompletas'}), 400

        username = input_validator.sanitize_input(data['username'])
        password = data['password']

        # Buscar usuario
        user = db.get_user_by_username(username)

        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'message': 'Credenciales inválidas'}), 401

        # Actualizar último login
        db.update_last_login(user['id'])

        # Generar token JWT
        token = jwt.encode({
            'user_id': user['id'],
            'username': user['username'],
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=app.config['JWT_EXPIRATION_HOURS'])
        }, app.config['SECRET_KEY'], algorithm='HS256')

        # Generar CSRF token
        csrf_token = csrf_protection.generate_token()

        return jsonify({
            'message': 'Login exitoso',
            'token': token,
            'csrf_token': csrf_token,
            'user': {
                'id': user['id'],
                'username': user['username'],
                'email': user['email']
            }
        }), 200

    except Exception as e:
        return jsonify({'message': f'Error en login: {str(e)}'}), 500

@app.route('/api/verify', methods=['GET'])
@token_required
def verify_token(current_user):
    """Verificar validez del token"""
    return jsonify({
        'message': 'Token válido',
        'user': {
            'id': current_user['id'],
            'username': current_user['username'],
            'email': current_user['email']
        }
    }), 200

# ==================== RUTAS DEL PROYECTO ====================

@app.route('/api/project/info', methods=['GET'])
@token_required
def get_project_info(current_user):
    """Obtener información general del proyecto"""
    project_info = {
        'title': 'Fine-Tuning Eficiente de LLaMA 3.2 (1B) para Español Latinoamericano',
        'subtitle': 'con QLoRA (4-bit) y Unsloth en Infraestructura HPC-UCR',
        'authors': [
            {
                'name': 'Alison Lobo Salas',
                'affiliation': 'Universidad de Costa Rica (UCR), CIOdD',
                'email': 'alison.lobo@ucr.ac.cr'
            },
            {
                'name': 'MSI. Agustín Gómez Meléndez',
                'affiliation': 'CIOdD--UCR',
                'email': 'agustin.gomez@ucr.ac.cr'
            }
        ],
        'date': 'Enero 2025',
        'abstract': 'Este informe técnico describe un flujo reproducible para adaptar el modelo de lenguaje grande meta-llama/Llama-3.2-1B-Instruct al español latinoamericano mediante fine-tuning eficiente con QLoRA (cuantización a 4 bits) y adaptadores LoRA, implementado con Unsloth y ejecutado en la infraestructura de computación de alto desempeño HPC-UCR (NVIDIA A100 80GB).',
        'key_contributions': [
            'Flujo completo y reproducible para ajuste fino eficiente mediante cuantización a 4 bits (QLoRA)',
            'Configuración centralizada mediante archivo config.py y ejecución reproducible',
            'Demostración de viabilidad técnica para entrenar modelos instructivos en español',
            'Resultados cuantitativos con análisis de convergencia',
            'Documentación completa del proceso',
            'Contribución al proyecto Latam-GPT'
        ]
    }
    return jsonify(project_info), 200

@app.route('/api/project/technical', methods=['GET'])
@token_required
def get_technical_details(current_user):
    """Obtener detalles técnicos del proyecto"""
    technical_details = {
        'model': {
            'name': 'meta-llama/Llama-3.2-1B-Instruct',
            'parameters': '1.24 mil millones',
            'architecture': 'Transformer decoder-only con atención causal',
            'context_length': '131,072 tokens',
            'tokenizer': 'BPE con vocabulario de ~128k tokens'
        },
        'training': {
            'technique': 'QLoRA (Quantized Low-Rank Adaptation)',
            'quantization': '4-bit (NF4)',
            'lora_rank': 16,
            'lora_alpha': 32,
            'learning_rate': 2e-4,
            'epochs': 60,
            'batch_size': 2,
            'gradient_accumulation': 4,
            'max_seq_length': 4096
        },
        'infrastructure': {
            'cluster': 'HPC-UCR',
            'gpu': 'NVIDIA A100 80GB',
            'training_time': '~18 horas (3 bloques × 6h)',
            'scheduler': 'SLURM'
        },
        'results': {
            'initial_perplexity': 21.74,
            'final_perplexity': 5.47,
            'perplexity_reduction': '74.8%',
            'initial_eval_loss': 3.08,
            'final_eval_loss': 1.70,
            'eval_loss_reduction': '44.8%'
        }
    }
    return jsonify(technical_details), 200

@app.route('/api/project/metrics', methods=['GET'])
@token_required
def get_metrics(current_user):
    """Obtener métricas de entrenamiento"""
    metrics = {
        'summary': {
            'train_loss': {'initial': 3.22, 'final': 0.14, 'reduction': '95.7%'},
            'eval_loss': {'initial': 3.08, 'final': 1.70, 'reduction': '44.8%'},
            'perplexity': {'initial': 21.74, 'final': 5.47, 'reduction': '74.8%'},
            'grad_norm': {'initial': 4.09, 'final': 0.29, 'reduction': '92.9%'}
        },
        'epoch_progress': [
            {'epoch': 1, 'train_loss': 3.2204, 'eval_loss': 3.0790, 'perplexity': 21.74, 'learning_rate': 2.66e-5, 'grad_norm': 4.09},
            {'epoch': 10, 'train_loss': 0.2142, 'eval_loss': 3.0143, 'perplexity': 20.37, 'learning_rate': 1.71e-4, 'grad_norm': 1.17},
            {'epoch': 20, 'train_loss': 0.1775, 'eval_loss': 2.9798, 'perplexity': 19.68, 'learning_rate': 1.37e-4, 'grad_norm': 1.07},
            {'epoch': 30, 'train_loss': 0.1523, 'eval_loss': 2.8126, 'perplexity': 16.65, 'learning_rate': 1.03e-4, 'grad_norm': 0.36},
            {'epoch': 40, 'train_loss': 0.1519, 'eval_loss': 2.6656, 'perplexity': 14.38, 'learning_rate': 6.88e-5, 'grad_norm': 0.33},
            {'epoch': 50, 'train_loss': 0.1454, 'eval_loss': 2.4571, 'perplexity': 11.67, 'learning_rate': 3.40e-5, 'grad_norm': 0.30},
            {'epoch': 60, 'train_loss': 0.1392, 'eval_loss': 1.7000, 'perplexity': 5.47, 'learning_rate': 4.35e-8, 'grad_norm': 0.29}
        ]
    }
    return jsonify(metrics), 200

@app.route('/api/project/infrastructure', methods=['GET'])
@token_required
def get_infrastructure(current_user):
    """Obtener detalles de infraestructura"""
    infrastructure = {
        'hpc': {
            'name': 'HPC-UCR',
            'location': 'Universidad de Costa Rica',
            'gpu_nodes': {
                'model': 'Lenovo ThinkSystem SR670 V2',
                'count': 2,
                'gpu_per_node': 4,
                'gpu_model': 'NVIDIA Tensor Core A100',
                'gpu_memory': '80GB',
                'total_gpu_memory': '320GB por nodo',
                'cpu_model': 'Intel Xeon Gold 6338',
                'cpu_cores': 64,
                'ram_per_core': '16GB'
            },
            'cpu_nodes': {
                'model': 'Lenovo ThinkSystem SD630 V2',
                'count': 16,
                'cpu_per_node': 'Intel Xeon Gold 6338',
                'cores_per_node': 64,
                'total_cores': 1024,
                'ram_per_core': '16GB'
            }
        },
        'software': {
            'os': 'Ubuntu 24.04 LTS',
            'python': '3.8+',
            'pytorch': '2.7.1',
            'cuda': '12.8',
            'transformers': 'Latest',
            'unsloth': 'Latest',
            'frameworks': ['PyTorch', 'HuggingFace Transformers', 'Unsloth', 'PEFT', 'BitsAndBytes']
        }
    }
    return jsonify(infrastructure), 200

@app.route('/api/project/methodology', methods=['GET'])
@token_required
def get_methodology(current_user):
    """Obtener detalles de metodología"""
    methodology = {
        'overview': 'Fine-tuning eficiente con QLoRA sobre LLaMA 3.2 1B Instruct',
        'techniques': {
            'qlora': {
                'description': 'Quantized Low-Rank Adaptation - combina cuantización 4-bit con adaptadores LoRA',
                'benefits': [
                    'Reducción drástica de memoria GPU',
                    'Mantiene calidad comparable a fine-tuning completo',
                    'Permite entrenar modelos grandes en hardware limitado'
                ],
                'components': [
                    'Cuantización NF4 (Normal Float 4)',
                    'Doble cuantización',
                    'Paginación optimizada'
                ]
            },
            'lora': {
                'description': 'Low-Rank Adaptation - entrena solo matrices de bajo rango',
                'parameters': {
                    'rank': 16,
                    'alpha': 32,
                    'dropout': 0.05,
                    'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
                }
            },
            'optimization': {
                'optimizer': 'AdamW',
                'scheduler': 'Warmup + Cosine Decay',
                'warmup_ratio': 0.03,
                'weight_decay': 0.01,
                'max_grad_norm': 1.0
            }
        },
        'reproducibility': {
            'seed': 42,
            'deterministic': True,
            'checkpointing': 'Cada 200 pasos',
            'evaluation': 'Cada 200 pasos'
        }
    }
    return jsonify(methodology), 200

@app.route('/api/project/collaboration', methods=['GET'])
@token_required
def get_collaboration(current_user):
    """Obtener información sobre colaboración con Latam-GPT"""
    collaboration = {
        'project': 'Latam-GPT',
        'organization': 'Centro Nacional de Inteligencia Artificial (CENIA) - Chile',
        'objective': 'Desarrollar un modelo fundacional de lenguaje de código abierto para América Latina',
        'contributions': [
            'Validación de metodologías de entrenamiento',
            'Documentación de experiencias técnicas',
            'Exploración de adaptación regional',
            'Contribución a corpus en español costarricense'
        ],
        'regional_impact': [
            'Soberanía tecnológica y lingüística',
            'Desarrollo de capacidades locales en IA',
            'Colaboración inter-institucional latinoamericana',
            'Democratización del acceso a LLMs contextualizados'
        ]
    }
    return jsonify(collaboration), 200

# ==================== RUTAS ESTÁTICAS ====================

@app.route('/')
def serve_frontend():
    """Servir página principal"""
    return send_from_directory('../frontend', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    """Servir archivos estáticos"""
    return send_from_directory('../frontend', path)

# ==================== MANEJO DE ERRORES ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'message': 'Recurso no encontrado'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'message': 'Error interno del servidor'}), 500

@app.errorhandler(429)
def ratelimit_handler(error):
    return jsonify({'message': 'Demasiadas solicitudes. Por favor intente más tarde.'}), 429

# ==================== PUNTO DE ENTRADA ====================

if __name__ == '__main__':
    # Inicializar base de datos
    db.init_db()

    # Configurar para desarrollo
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    port = int(os.environ.get('PORT', 5000))

    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║  LLM-HPC Project Information Portal                       ║
    ║  Backend Flask con Seguridad y Autenticación              ║
    ╠════════════════════════════════════════════════════════════╣
    ║  Servidor ejecutándose en: http://localhost:{port}        ║
    ║  Modo: {'Desarrollo' if debug_mode else 'Producción'}     ║
    ╚════════════════════════════════════════════════════════════╝
    """)

    app.run(host='0.0.0.0', port=port, debug=debug_mode)
