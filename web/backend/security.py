"""
Módulo de seguridad y ciberseguridad
Implementa protecciones contra ataques comunes
"""

import secrets
import hashlib
import re
from datetime import datetime, timedelta
from typing import Optional

class SecurityHeaders:
    """Gestión de headers de seguridad HTTP"""

    @staticmethod
    def add_headers(response):
        """
        Añade headers de seguridad a la respuesta HTTP
        Protección contra XSS, Clickjacking, MIME sniffing, etc.
        """
        # Content Security Policy - Previene XSS
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "connect-src 'self'"
        )

        # X-Content-Type-Options - Previene MIME sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'

        # X-Frame-Options - Previene clickjacking
        response.headers['X-Frame-Options'] = 'DENY'

        # X-XSS-Protection - Protección XSS legacy
        response.headers['X-XSS-Protection'] = '1; mode=block'

        # Strict-Transport-Security - Fuerza HTTPS (cuando se use HTTPS)
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'

        # Referrer-Policy - Control de información de referrer
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'

        # Permissions-Policy - Control de features del navegador
        response.headers['Permissions-Policy'] = (
            'geolocation=(), '
            'microphone=(), '
            'camera=(), '
            'payment=(), '
            'usb=(), '
            'magnetometer=(), '
            'gyroscope=()'
        )

        return response


class InputValidator:
    """Validación y sanitización de entradas de usuario"""

    # Expresiones regulares para validación
    USERNAME_REGEX = re.compile(r'^[a-zA-Z0-9_]{3,20}$')
    EMAIL_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    PASSWORD_REGEX = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)[a-zA-Z\d@$!%*?&]{8,}$')

    @classmethod
    def validate_username(cls, username: str) -> bool:
        """
        Valida formato de nombre de usuario
        - Solo alfanuméricos y guión bajo
        - Entre 3 y 20 caracteres
        """
        if not username or not isinstance(username, str):
            return False
        return bool(cls.USERNAME_REGEX.match(username))

    @classmethod
    def validate_email(cls, email: str) -> bool:
        """
        Valida formato de email
        """
        if not email or not isinstance(email, str):
            return False
        return bool(cls.EMAIL_REGEX.match(email))

    @classmethod
    def validate_password(cls, password: str) -> bool:
        """
        Valida fortaleza de contraseña
        - Mínimo 8 caracteres
        - Al menos una mayúscula
        - Al menos una minúscula
        - Al menos un número
        """
        if not password or not isinstance(password, str):
            return False
        return bool(cls.PASSWORD_REGEX.match(password))

    @staticmethod
    def sanitize_input(input_str: str) -> str:
        """
        Sanitiza entrada de usuario para prevenir inyecciones
        - Elimina caracteres peligrosos
        - Escapa caracteres especiales
        """
        if not input_str:
            return ""

        # Eliminar caracteres de control
        sanitized = ''.join(char for char in input_str if ord(char) >= 32)

        # Limitar longitud
        sanitized = sanitized[:200]

        # Eliminar secuencias peligrosas
        dangerous_patterns = [
            '<script', '</script', 'javascript:', 'onerror=', 'onclick=',
            'onload=', '<iframe', '</iframe', 'eval(', 'setTimeout(',
            'setInterval(', 'Function(', '--', ';--', '/*', '*/',
            'exec(', 'system(', '__import__'
        ]

        sanitized_lower = sanitized.lower()
        for pattern in dangerous_patterns:
            if pattern in sanitized_lower:
                sanitized = sanitized.replace(pattern, '')

        return sanitized.strip()

    @staticmethod
    def validate_sql_injection(input_str: str) -> bool:
        """
        Detecta posibles intentos de SQL injection
        Retorna True si la entrada parece segura
        """
        if not input_str:
            return True

        # Patrones comunes de SQL injection
        sql_patterns = [
            r"(\bunion\b.*\bselect\b)",
            r"(\bselect\b.*\bfrom\b)",
            r"(\binsert\b.*\binto\b)",
            r"(\bupdate\b.*\bset\b)",
            r"(\bdelete\b.*\bfrom\b)",
            r"(\bdrop\b.*\btable\b)",
            r"(;.*--)",
            r"(--.*$)",
            r"(/\*.*\*/)",
            r"(exec\()",
            r"(execute\()",
            r"(xp_.*\()"
        ]

        input_lower = input_str.lower()
        for pattern in sql_patterns:
            if re.search(pattern, input_lower, re.IGNORECASE):
                return False

        return True

    @staticmethod
    def validate_xss(input_str: str) -> bool:
        """
        Detecta posibles intentos de XSS
        Retorna True si la entrada parece segura
        """
        if not input_str:
            return True

        # Patrones comunes de XSS
        xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'onerror\s*=',
            r'onload\s*=',
            r'onclick\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
            r'eval\s*\(',
            r'expression\s*\('
        ]

        for pattern in xss_patterns:
            if re.search(pattern, input_str, re.IGNORECASE):
                return False

        return True


class CSRFProtection:
    """Protección contra Cross-Site Request Forgery"""

    def __init__(self):
        self.tokens = {}  # En producción, usar Redis o base de datos
        self.token_lifetime = timedelta(hours=1)

    def generate_token(self) -> str:
        """
        Genera un token CSRF único
        """
        token = secrets.token_urlsafe(32)
        self.tokens[token] = datetime.utcnow()
        return token

    def validate_token(self, token: str) -> bool:
        """
        Valida un token CSRF
        """
        if not token or token not in self.tokens:
            return False

        # Verificar expiración
        token_time = self.tokens[token]
        if datetime.utcnow() - token_time > self.token_lifetime:
            del self.tokens[token]
            return False

        return True

    def invalidate_token(self, token: str):
        """
        Invalida un token CSRF usado
        """
        if token in self.tokens:
            del self.tokens[token]

    def cleanup_expired(self):
        """
        Limpia tokens expirados
        """
        now = datetime.utcnow()
        expired = [
            token for token, time in self.tokens.items()
            if now - time > self.token_lifetime
        ]
        for token in expired:
            del self.tokens[token]


class PasswordPolicy:
    """Política de contraseñas seguras"""

    @staticmethod
    def check_strength(password: str) -> dict:
        """
        Evalúa la fortaleza de una contraseña
        Retorna un diccionario con score y sugerencias
        """
        score = 0
        suggestions = []

        # Longitud
        if len(password) < 8:
            suggestions.append("Use al menos 8 caracteres")
        elif len(password) >= 12:
            score += 2
        else:
            score += 1

        # Mayúsculas
        if not re.search(r'[A-Z]', password):
            suggestions.append("Incluya al menos una letra mayúscula")
        else:
            score += 1

        # Minúsculas
        if not re.search(r'[a-z]', password):
            suggestions.append("Incluya al menos una letra minúscula")
        else:
            score += 1

        # Números
        if not re.search(r'\d', password):
            suggestions.append("Incluya al menos un número")
        else:
            score += 1

        # Caracteres especiales
        if re.search(r'[@$!%*?&#]', password):
            score += 2
        else:
            suggestions.append("Considere usar caracteres especiales (@$!%*?&#)")

        # Patrones comunes débiles
        weak_patterns = ['123', 'abc', 'qwerty', 'password', 'admin', '000']
        if any(pattern in password.lower() for pattern in weak_patterns):
            score -= 2
            suggestions.append("Evite patrones comunes (123, abc, password, etc.)")

        # Determinar nivel
        if score >= 6:
            strength = "Fuerte"
        elif score >= 4:
            strength = "Media"
        else:
            strength = "Débil"

        return {
            'score': max(0, score),
            'strength': strength,
            'suggestions': suggestions
        }

    @staticmethod
    def is_commonly_used(password: str) -> bool:
        """
        Verifica si la contraseña está en una lista de contraseñas comunes
        """
        # Top 100 contraseñas más comunes (simplificado)
        common_passwords = {
            'password', '123456', '123456789', '12345678', '12345', '1234567',
            'password1', 'qwerty', 'abc123', '111111', 'monkey', '1234567890',
            'letmein', 'trustno1', 'dragon', 'baseball', 'iloveyou', 'master',
            'sunshine', 'ashley', 'bailey', 'passw0rd', 'shadow', '123123',
            '654321', 'superman', 'qazwsx', 'michael', 'football', 'admin'
        }

        return password.lower() in common_passwords


class RateLimitTracker:
    """
    Rastreador de rate limiting por IP
    En producción, usar Redis para compartir entre instancias
    """

    def __init__(self):
        self.requests = {}  # IP -> [(timestamp, endpoint)]
        self.cleanup_interval = 300  # 5 minutos
        self.last_cleanup = datetime.utcnow()

    def is_rate_limited(self, ip: str, endpoint: str, limit: int, window: int) -> bool:
        """
        Verifica si una IP ha excedido el límite de rate

        Args:
            ip: Dirección IP
            endpoint: Endpoint solicitado
            limit: Número máximo de requests
            window: Ventana de tiempo en segundos

        Returns:
            True si está limitado, False si puede continuar
        """
        now = datetime.utcnow()

        # Cleanup periódico
        if (now - self.last_cleanup).seconds > self.cleanup_interval:
            self.cleanup_old_requests()

        # Inicializar si es nueva IP
        if ip not in self.requests:
            self.requests[ip] = []

        # Filtrar requests dentro de la ventana de tiempo
        cutoff_time = now - timedelta(seconds=window)
        recent_requests = [
            (ts, ep) for ts, ep in self.requests[ip]
            if ts > cutoff_time and ep == endpoint
        ]

        # Verificar límite
        if len(recent_requests) >= limit:
            return True

        # Registrar nuevo request
        self.requests[ip].append((now, endpoint))

        return False

    def cleanup_old_requests(self):
        """Limpia requests antiguos"""
        cutoff = datetime.utcnow() - timedelta(seconds=3600)  # 1 hora

        for ip in list(self.requests.keys()):
            self.requests[ip] = [
                (ts, ep) for ts, ep in self.requests[ip]
                if ts > cutoff
            ]

            # Eliminar IPs sin requests recientes
            if not self.requests[ip]:
                del self.requests[ip]

        self.last_cleanup = datetime.utcnow()


class SecurityAudit:
    """Registro de eventos de seguridad para auditoría"""

    def __init__(self):
        self.events = []

    def log_event(self, event_type: str, details: dict, severity: str = "INFO"):
        """
        Registra un evento de seguridad

        Args:
            event_type: Tipo de evento (login_failed, token_invalid, etc.)
            details: Detalles del evento
            severity: Nivel de severidad (INFO, WARNING, ERROR, CRITICAL)
        """
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'type': event_type,
            'severity': severity,
            'details': details
        }

        self.events.append(event)

        # En producción, escribir a archivo de log o sistema de logging
        print(f"[SECURITY {severity}] {event_type}: {details}")

    def get_recent_events(self, hours: int = 24, severity: Optional[str] = None):
        """Obtiene eventos recientes"""
        cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat()

        events = [e for e in self.events if e['timestamp'] > cutoff]

        if severity:
            events = [e for e in events if e['severity'] == severity]

        return events
