# Portal Web de Información del Proyecto LLM-HPC

Portal web seguro con autenticación y componentes de ciberseguridad para presentar la información detallada del proyecto de Fine-Tuning de LLaMA 3.2 para Español Latinoamericano.

## 🎯 Características Principales

### 📊 Funcionalidades
- **Información Completa del Proyecto**: Presenta todos los detalles técnicos, metodología, resultados y colaboraciones
- **Navegación Intuitiva**: Interfaz moderna con navegación suave entre secciones
- **Diseño Responsive**: Adaptado para desktop, tablet y móvil
- **Visualización de Métricas**: Tablas interactivas con resultados de entrenamiento
- **Información de Infraestructura**: Detalles del cluster HPC-UCR

### 🔒 Seguridad y Ciberseguridad

El portal implementa múltiples capas de seguridad:

#### Autenticación y Autorización
- ✅ **JWT (JSON Web Tokens)**: Autenticación basada en tokens con expiración
- ✅ **Sesiones Seguras**: Gestión de sesiones de usuario
- ✅ **Hash de Contraseñas**: PBKDF2-SHA256 para cifrado de contraseñas

#### Protecciones contra Ataques
- ✅ **CSRF Protection**: Tokens CSRF para prevenir ataques Cross-Site Request Forgery
- ✅ **XSS Prevention**: Validación y sanitización de entradas
- ✅ **SQL Injection Prevention**: Validación de patrones SQL
- ✅ **Rate Limiting**: Límites de solicitudes por IP para prevenir fuerza bruta
- ✅ **Headers de Seguridad HTTP**:
  - Content-Security-Policy
  - X-Content-Type-Options
  - X-Frame-Options
  - X-XSS-Protection
  - Strict-Transport-Security
  - Referrer-Policy
  - Permissions-Policy

#### Validación de Datos
- ✅ **Validación de Formato**: Username, email, contraseñas
- ✅ **Política de Contraseñas Fuertes**: Mínimo 8 caracteres, mayúsculas, minúsculas y números
- ✅ **Sanitización de Entradas**: Eliminación de caracteres peligrosos
- ✅ **Detección de Patrones Maliciosos**: Identificación de intentos de inyección

## 📁 Estructura del Proyecto

```
web/
├── backend/                    # Backend Flask
│   ├── app.py                 # Aplicación principal con API REST
│   ├── security.py            # Módulo de seguridad y ciberseguridad
│   ├── models.py              # Modelos de base de datos
│   ├── requirements.txt       # Dependencias Python
│   └── llm_hpc.db            # Base de datos SQLite (generada)
│
├── frontend/                  # Frontend HTML/CSS/JS
│   ├── index.html            # Página principal del portal
│   ├── login.html            # Página de autenticación
│   ├── css/
│   │   └── styles.css        # Estilos CSS responsive
│   └── js/
│       ├── app.js            # Lógica principal de la aplicación
│       └── auth.js           # Lógica de autenticación
│
└── README.md                  # Esta documentación
```

## 🚀 Instalación y Configuración

### Requisitos Previos

- Python 3.8+
- pip (gestor de paquetes de Python)
- Navegador web moderno

### Paso 1: Instalar Dependencias del Backend

```bash
cd web/backend
pip install -r requirements.txt
```

### Paso 2: Configurar Variables de Entorno (Opcional)

Para producción, configure una clave secreta personalizada:

```bash
export SECRET_KEY="tu_clave_secreta_muy_segura_aqui"
export FLASK_ENV="production"
```

Para desarrollo:

```bash
export FLASK_ENV="development"
```

### Paso 3: Inicializar la Base de Datos

La base de datos se inicializa automáticamente al ejecutar la aplicación por primera vez.

## 🏃 Ejecución

### Modo Desarrollo

```bash
cd web/backend
python app.py
```

La aplicación estará disponible en: http://localhost:5000

### Modo Producción

Para producción, se recomienda usar un servidor WSGI como Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

O con uWSGI:

```bash
pip install uwsgi
uwsgi --http :5000 --wsgi-file app.py --callable app --processes 4
```

## 🔐 Uso del Portal

### 1. Registro de Usuario

1. Acceder a http://localhost:5000/login.html
2. Hacer clic en la pestaña "Registrarse"
3. Completar el formulario:
   - **Usuario**: 3-20 caracteres alfanuméricos
   - **Email**: Dirección de correo válida
   - **Contraseña**: Mínimo 8 caracteres con mayúsculas, minúsculas y números
4. Confirmar contraseña
5. Hacer clic en "Registrarse"

### 2. Inicio de Sesión

1. En la pestaña "Iniciar Sesión"
2. Ingresar usuario y contraseña
3. Hacer clic en "Ingresar"
4. Serás redirigido al portal principal

### 3. Navegación en el Portal

El portal incluye las siguientes secciones:

- **Resumen**: Abstract y contribuciones principales
- **Técnico**: Detalles del modelo y configuración de entrenamiento
- **Resultados**: Métricas y progreso de entrenamiento
- **Infraestructura**: Información del cluster HPC-UCR
- **Metodología**: Técnicas utilizadas (QLoRA, LoRA, optimización)
- **Colaboración**: Información sobre Latam-GPT

### 4. Cerrar Sesión

Hacer clic en el botón "Cerrar Sesión" en la parte superior derecha.

## 🛡️ Configuración de Seguridad

### Rate Limiting

El backend implementa límites de tasa por defecto:

- **General**: 200 solicitudes/día, 50 solicitudes/hora
- **Login**: 10 intentos/minuto
- **Registro**: 5 intentos/hora

Para modificar estos límites, editar en `app.py`:

```python
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)
```

### Política de Contraseñas

La política de contraseñas puede ajustarse en `security.py`:

```python
PASSWORD_REGEX = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&]{8,}$')
```

Requisitos actuales:
- Mínimo 8 caracteres
- Al menos una mayúscula
- Al menos una minúscula
- Al menos un número
- Opcionalmente caracteres especiales

### Expiración de JWT

Por defecto, los tokens JWT expiran en 24 horas. Para modificar:

```python
app.config['JWT_EXPIRATION_HOURS'] = 24  # Cambiar valor aquí
```

## 📊 API REST Endpoints

### Autenticación

- `POST /api/register` - Registro de nuevo usuario
- `POST /api/login` - Inicio de sesión
- `GET /api/verify` - Verificar token JWT

### Información del Proyecto

- `GET /api/project/info` - Información general
- `GET /api/project/technical` - Detalles técnicos
- `GET /api/project/metrics` - Métricas de entrenamiento
- `GET /api/project/infrastructure` - Información de infraestructura
- `GET /api/project/methodology` - Metodología utilizada
- `GET /api/project/collaboration` - Colaboración con Latam-GPT

Todos los endpoints (excepto `/register` y `/login`) requieren autenticación JWT:

```javascript
fetch('http://localhost:5000/api/project/info', {
    headers: {
        'Authorization': 'Bearer ' + jwt_token
    }
})
```

## 🗄️ Base de Datos

### Tablas

- **users**: Información de usuarios
- **sessions**: Sesiones activas
- **login_attempts**: Intentos de login para detección de ataques
- **security_audit**: Registro de eventos de seguridad

### Backup

Realizar backup de la base de datos:

```bash
cp web/backend/llm_hpc.db web/backend/llm_hpc_backup_$(date +%Y%m%d).db
```

## 🧪 Testing

### Pruebas Manuales

1. **Registro**: Crear usuario con diferentes combinaciones de datos
2. **Login**: Probar credenciales correctas e incorrectas
3. **Rate Limiting**: Intentar múltiples logins rápidos
4. **XSS**: Intentar inyectar scripts en formularios
5. **SQL Injection**: Intentar inyección SQL en campos de texto
6. **CSRF**: Verificar tokens CSRF en requests

### Pruebas Automatizadas (Recomendadas)

Crear archivo `test_security.py`:

```python
import pytest
from security import InputValidator, PasswordPolicy

def test_username_validation():
    assert InputValidator.validate_username("user123")
    assert not InputValidator.validate_username("us")
    assert not InputValidator.validate_username("user<script>")

def test_password_strength():
    result = PasswordPolicy.check_strength("Password123")
    assert result['strength'] == 'Fuerte'
```

Ejecutar:

```bash
pytest test_security.py
```

## 🔧 Troubleshooting

### Error: "Port 5000 already in use"

Cambiar el puerto en `app.py`:

```python
port = int(os.environ.get('PORT', 8080))  # Usar puerto 8080
```

### Error: "Token inválido"

Limpiar localStorage del navegador:

```javascript
localStorage.clear()
```

### Error: "CORS policy"

Verificar configuración CORS en `app.py` y ajustar origins:

```python
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5000", "tu_dominio_aqui"]
    }
})
```

## 📝 Logs y Monitoreo

Los eventos de seguridad se registran en:

1. **Consola del servidor**: Eventos en tiempo real
2. **Base de datos**: Tabla `security_audit`
3. **Logs de SLURM** (si se ejecuta en HPC): `logs/*.out`

Ver eventos recientes de seguridad:

```python
from models import Database

db = Database()
events = db.get_security_events(hours=24, severity='CRITICAL')
for event in events:
    print(event)
```

## 🌐 Despliegue en Producción

### Consideraciones

1. **HTTPS**: Usar certificado SSL/TLS (Let's Encrypt)
2. **Reverse Proxy**: Nginx o Apache como proxy
3. **Firewall**: Configurar firewall para limitar acceso
4. **Secrets**: No hardcodear claves secretas
5. **Base de Datos**: Migrar a PostgreSQL o MySQL para producción
6. **Backup**: Implementar backups automáticos
7. **Monitoring**: Configurar alertas de seguridad

### Ejemplo Nginx

```nginx
server {
    listen 80;
    server_name tu-dominio.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 🤝 Contribuciones

Para contribuir al proyecto:

1. Fork del repositorio
2. Crear rama para feature: `git checkout -b feature/nueva-funcionalidad`
3. Commit de cambios: `git commit -m 'Agregar nueva funcionalidad'`
4. Push a la rama: `git push origin feature/nueva-funcionalidad`
5. Crear Pull Request

## 📄 Licencia

Este proyecto forma parte del repositorio LLM-HPC y sigue la misma licencia del proyecto principal.

## 📧 Contacto

- **Alison Lobo Salas**: alison.lobo@ucr.ac.cr
- **MSI. Agustín Gómez Meléndez**: agustin.gomez@ucr.ac.cr

## 🙏 Agradecimientos

- Universidad de Costa Rica (UCR) - CIOdD
- Proyecto Latam-GPT (CENIA, Chile)
- HPC-UCR por la infraestructura

---

**Nota de Seguridad**: Este portal implementa múltiples capas de seguridad, pero siempre se recomienda realizar auditorías de seguridad periódicas y mantener todas las dependencias actualizadas.
