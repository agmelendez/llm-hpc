"""
Modelos de base de datos para usuarios y sesiones
Utiliza SQLite para simplicidad
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, Dict, List
from contextlib import contextmanager

class Database:
    """Gestor de base de datos SQLite"""

    def __init__(self, db_path: str = 'llm_hpc.db'):
        """
        Inicializa la conexión a la base de datos

        Args:
            db_path: Ruta al archivo de base de datos
        """
        # Usar ruta absoluta en directorio backend
        backend_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(backend_dir, db_path)

    @contextmanager
    def get_connection(self):
        """Context manager para conexiones de base de datos"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Permite acceso por nombre de columna
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def init_db(self):
        """Inicializa las tablas de la base de datos"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Tabla de usuarios
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    is_admin BOOLEAN DEFAULT 0
                )
            ''')

            # Índices para búsqueda rápida
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_username ON users(username)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_email ON users(email)
            ''')

            # Tabla de sesiones
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    token_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    ip_address TEXT,
                    user_agent TEXT,
                    is_valid BOOLEAN DEFAULT 1,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_token ON sessions(token_hash)
            ''')

            # Tabla de intentos de login fallidos (para detección de ataques)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS login_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    ip_address TEXT NOT NULL,
                    attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN NOT NULL
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_login_attempts
                ON login_attempts(username, attempt_time)
            ''')

            # Tabla de auditoría de seguridad
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS security_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    user_id INTEGER,
                    ip_address TEXT,
                    details TEXT,
                    severity TEXT DEFAULT 'INFO',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')

            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON security_audit(timestamp)
            ''')

            print(f"✓ Base de datos inicializada: {self.db_path}")

    # ==================== OPERACIONES DE USUARIOS ====================

    def create_user(self, username: str, password_hash: str, email: str) -> int:
        """
        Crea un nuevo usuario

        Args:
            username: Nombre de usuario
            password_hash: Hash de la contraseña
            email: Email del usuario

        Returns:
            ID del usuario creado
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO users (username, password_hash, email)
                VALUES (?, ?, ?)
            ''', (username, password_hash, email))

            user_id = cursor.lastrowid

            # Auditar creación
            self.log_security_event('user_created', user_id, None,
                                   f"Usuario {username} creado", 'INFO')

            return user_id

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """
        Obtiene un usuario por nombre de usuario

        Args:
            username: Nombre de usuario

        Returns:
            Diccionario con datos del usuario o None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, username, password_hash, email, created_at,
                       last_login, is_active, is_admin
                FROM users
                WHERE username = ? AND is_active = 1
            ''', (username,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """
        Obtiene un usuario por email

        Args:
            email: Email del usuario

        Returns:
            Diccionario con datos del usuario o None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, username, password_hash, email, created_at,
                       last_login, is_active, is_admin
                FROM users
                WHERE email = ? AND is_active = 1
            ''', (email,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        """
        Obtiene un usuario por ID

        Args:
            user_id: ID del usuario

        Returns:
            Diccionario con datos del usuario o None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, username, password_hash, email, created_at,
                       last_login, is_active, is_admin
                FROM users
                WHERE id = ? AND is_active = 1
            ''', (user_id,))

            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def update_last_login(self, user_id: int):
        """
        Actualiza el timestamp del último login

        Args:
            user_id: ID del usuario
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users
                SET last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (user_id,))

    def deactivate_user(self, user_id: int):
        """
        Desactiva un usuario (soft delete)

        Args:
            user_id: ID del usuario
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE users
                SET is_active = 0
                WHERE id = ?
            ''', (user_id,))

            self.log_security_event('user_deactivated', user_id, None,
                                   f"Usuario {user_id} desactivado", 'WARNING')

    def get_all_users(self, include_inactive: bool = False) -> List[Dict]:
        """
        Obtiene todos los usuarios

        Args:
            include_inactive: Si incluir usuarios inactivos

        Returns:
            Lista de usuarios
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if include_inactive:
                cursor.execute('''
                    SELECT id, username, email, created_at, last_login,
                           is_active, is_admin
                    FROM users
                ''')
            else:
                cursor.execute('''
                    SELECT id, username, email, created_at, last_login,
                           is_active, is_admin
                    FROM users
                    WHERE is_active = 1
                ''')

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # ==================== OPERACIONES DE SESIONES ====================

    def create_session(self, user_id: int, token_hash: str, expires_at: datetime,
                      ip_address: str = None, user_agent: str = None) -> int:
        """
        Crea una nueva sesión

        Args:
            user_id: ID del usuario
            token_hash: Hash del token de sesión
            expires_at: Timestamp de expiración
            ip_address: IP del cliente
            user_agent: User agent del cliente

        Returns:
            ID de la sesión creada
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (user_id, token_hash, expires_at,
                                    ip_address, user_agent)
                VALUES (?, ?, ?, ?, ?)
            ''', (user_id, token_hash, expires_at, ip_address, user_agent))

            return cursor.lastrowid

    def invalidate_session(self, token_hash: str):
        """
        Invalida una sesión

        Args:
            token_hash: Hash del token
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE sessions
                SET is_valid = 0
                WHERE token_hash = ?
            ''', (token_hash,))

    # ==================== INTENTOS DE LOGIN ====================

    def log_login_attempt(self, username: str, ip_address: str, success: bool):
        """
        Registra un intento de login

        Args:
            username: Nombre de usuario
            ip_address: IP del intento
            success: Si fue exitoso
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO login_attempts (username, ip_address, success)
                VALUES (?, ?, ?)
            ''', (username, ip_address, success))

    def get_failed_login_attempts(self, username: str, minutes: int = 15) -> int:
        """
        Cuenta intentos fallidos recientes

        Args:
            username: Nombre de usuario
            minutes: Ventana de tiempo en minutos

        Returns:
            Número de intentos fallidos
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COUNT(*)
                FROM login_attempts
                WHERE username = ?
                  AND success = 0
                  AND attempt_time > datetime('now', '-' || ? || ' minutes')
            ''', (username, minutes))

            return cursor.fetchone()[0]

    # ==================== AUDITORÍA DE SEGURIDAD ====================

    def log_security_event(self, event_type: str, user_id: Optional[int],
                          ip_address: Optional[str], details: str,
                          severity: str = 'INFO'):
        """
        Registra un evento de seguridad

        Args:
            event_type: Tipo de evento
            user_id: ID del usuario (si aplica)
            ip_address: IP del evento
            details: Detalles del evento
            severity: Nivel de severidad
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO security_audit
                (event_type, user_id, ip_address, details, severity)
                VALUES (?, ?, ?, ?, ?)
            ''', (event_type, user_id, ip_address, details, severity))

    def get_security_events(self, hours: int = 24,
                           severity: Optional[str] = None) -> List[Dict]:
        """
        Obtiene eventos de seguridad recientes

        Args:
            hours: Ventana de tiempo en horas
            severity: Filtrar por severidad

        Returns:
            Lista de eventos
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            if severity:
                cursor.execute('''
                    SELECT * FROM security_audit
                    WHERE timestamp > datetime('now', '-' || ? || ' hours')
                      AND severity = ?
                    ORDER BY timestamp DESC
                ''', (hours, severity))
            else:
                cursor.execute('''
                    SELECT * FROM security_audit
                    WHERE timestamp > datetime('now', '-' || ? || ' hours')
                    ORDER BY timestamp DESC
                ''', (hours,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # ==================== ESTADÍSTICAS ====================

    def get_user_stats(self) -> Dict:
        """Obtiene estadísticas de usuarios"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 1')
            active_users = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM users WHERE is_active = 0')
            inactive_users = cursor.fetchone()[0]

            cursor.execute('''
                SELECT COUNT(*) FROM users
                WHERE last_login > datetime('now', '-7 days')
            ''')
            active_this_week = cursor.fetchone()[0]

            return {
                'total_active': active_users,
                'total_inactive': inactive_users,
                'active_this_week': active_this_week
            }

    def get_security_stats(self) -> Dict:
        """Obtiene estadísticas de seguridad"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Intentos fallidos últimas 24h
            cursor.execute('''
                SELECT COUNT(*) FROM login_attempts
                WHERE success = 0
                  AND attempt_time > datetime('now', '-1 day')
            ''')
            failed_logins_24h = cursor.fetchone()[0]

            # Eventos críticos últimas 24h
            cursor.execute('''
                SELECT COUNT(*) FROM security_audit
                WHERE severity = 'CRITICAL'
                  AND timestamp > datetime('now', '-1 day')
            ''')
            critical_events_24h = cursor.fetchone()[0]

            return {
                'failed_logins_24h': failed_logins_24h,
                'critical_events_24h': critical_events_24h
            }
