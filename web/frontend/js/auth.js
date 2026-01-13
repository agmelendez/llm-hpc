/**
 * Módulo de Autenticación
 * Maneja login, registro y gestión de tokens JWT
 */

const API_BASE_URL = 'http://localhost:5000/api';

// ==================== UTILIDADES ====================

/**
 * Muestra una alerta en la interfaz
 */
function showAlert(message, type = 'info') {
    const alertContainer = document.getElementById('alert-container');
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;

    const icon = {
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'info': 'ℹ️'
    }[type] || 'ℹ️';

    alert.innerHTML = `<strong>${icon}</strong> ${message}`;
    alertContainer.innerHTML = '';
    alertContainer.appendChild(alert);

    // Auto-ocultar después de 5 segundos
    setTimeout(() => {
        alert.style.opacity = '0';
        setTimeout(() => alert.remove(), 300);
    }, 5000);
}

/**
 * Cambia entre tabs de login y registro
 */
function showTab(tab) {
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    const tabs = document.querySelectorAll('.tab-button');

    tabs.forEach(btn => btn.classList.remove('active'));

    if (tab === 'login') {
        loginForm.style.display = 'block';
        registerForm.style.display = 'none';
        tabs[0].classList.add('active');
    } else {
        loginForm.style.display = 'none';
        registerForm.style.display = 'block';
        tabs[1].classList.add('active');
    }

    // Limpiar alertas
    document.getElementById('alert-container').innerHTML = '';
}

/**
 * Valida la fortaleza de la contraseña en tiempo real
 */
function checkPasswordStrength(password) {
    const strengthDiv = document.getElementById('password-strength');
    if (!strengthDiv) return;

    let score = 0;
    let feedback = [];

    // Longitud
    if (password.length >= 8) score++;
    if (password.length >= 12) score++;

    // Complejidad
    if (/[a-z]/.test(password)) score++;
    if (/[A-Z]/.test(password)) score++;
    if (/\d/.test(password)) score++;
    if (/[@$!%*?&#]/.test(password)) score++;

    // Patrones débiles
    const weakPatterns = ['123', 'abc', 'password', 'qwerty'];
    if (weakPatterns.some(p => password.toLowerCase().includes(p))) {
        score -= 2;
    }

    // Determinar nivel
    let level, color, text;
    if (score >= 6) {
        level = 'strong';
        color = '#4caf50';
        text = '✅ Fuerte';
    } else if (score >= 4) {
        level = 'medium';
        color = '#ff9800';
        text = '⚠️ Media';
    } else {
        level = 'weak';
        color = '#f44336';
        text = '❌ Débil';
    }

    strengthDiv.innerHTML = `
        <div class="strength-bar">
            <div class="strength-fill strength-${level}" style="width: ${(score / 6) * 100}%; background-color: ${color};"></div>
        </div>
        <span style="color: ${color};">${text}</span>
    `;
}

// ==================== API CALLS ====================

/**
 * Realiza login
 */
async function login(username, password) {
    try {
        const response = await fetch(`${API_BASE_URL}/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();

        if (response.ok) {
            // Guardar token y datos de usuario
            localStorage.setItem('jwt_token', data.token);
            localStorage.setItem('csrf_token', data.csrf_token);
            localStorage.setItem('user', JSON.stringify(data.user));

            showAlert('Login exitoso. Redirigiendo...', 'success');

            // Redirigir a página principal
            setTimeout(() => {
                window.location.href = 'index.html';
            }, 1000);
        } else {
            showAlert(data.message || 'Error en login', 'error');
        }
    } catch (error) {
        console.error('Error en login:', error);
        showAlert('Error de conexión. Por favor intente de nuevo.', 'error');
    }
}

/**
 * Realiza registro
 */
async function register(username, email, password) {
    try {
        const response = await fetch(`${API_BASE_URL}/register`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username, email, password })
        });

        const data = await response.json();

        if (response.ok) {
            showAlert('Registro exitoso. Ya puede iniciar sesión.', 'success');

            // Cambiar a tab de login
            setTimeout(() => {
                showTab('login');
                document.getElementById('login-username').value = username;
            }, 1500);
        } else {
            showAlert(data.message || 'Error en registro', 'error');
        }
    } catch (error) {
        console.error('Error en registro:', error);
        showAlert('Error de conexión. Por favor intente de nuevo.', 'error');
    }
}

// ==================== EVENT LISTENERS ====================

document.addEventListener('DOMContentLoaded', () => {
    // Verificar si ya está autenticado
    const token = localStorage.getItem('jwt_token');
    if (token) {
        // Verificar validez del token
        fetch(`${API_BASE_URL}/verify`, {
            headers: {
                'Authorization': `Bearer ${token}`
            }
        })
        .then(response => {
            if (response.ok) {
                // Token válido, redirigir a página principal
                window.location.href = 'index.html';
            } else {
                // Token inválido, limpiar storage
                localStorage.clear();
            }
        })
        .catch(() => {
            localStorage.clear();
        });
    }

    // Form de login
    const loginForm = document.getElementById('login-form');
    loginForm.addEventListener('submit', (e) => {
        e.preventDefault();

        const username = document.getElementById('login-username').value.trim();
        const password = document.getElementById('login-password').value;

        if (!username || !password) {
            showAlert('Por favor complete todos los campos', 'warning');
            return;
        }

        login(username, password);
    });

    // Form de registro
    const registerForm = document.getElementById('register-form');
    registerForm.addEventListener('submit', (e) => {
        e.preventDefault();

        const username = document.getElementById('register-username').value.trim();
        const email = document.getElementById('register-email').value.trim();
        const password = document.getElementById('register-password').value;
        const passwordConfirm = document.getElementById('register-password-confirm').value;

        // Validaciones
        if (!username || !email || !password || !passwordConfirm) {
            showAlert('Por favor complete todos los campos', 'warning');
            return;
        }

        // Validar formato de usuario
        if (!/^[a-zA-Z0-9_]{3,20}$/.test(username)) {
            showAlert('Usuario debe tener 3-20 caracteres alfanuméricos', 'error');
            return;
        }

        // Validar email
        if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
            showAlert('Email inválido', 'error');
            return;
        }

        // Validar contraseña
        if (password.length < 8) {
            showAlert('Contraseña debe tener al menos 8 caracteres', 'error');
            return;
        }

        if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(password)) {
            showAlert('Contraseña debe incluir mayúsculas, minúsculas y números', 'error');
            return;
        }

        // Validar coincidencia
        if (password !== passwordConfirm) {
            showAlert('Las contraseñas no coinciden', 'error');
            return;
        }

        register(username, email, password);
    });

    // Validación de fortaleza de contraseña en tiempo real
    const registerPassword = document.getElementById('register-password');
    if (registerPassword) {
        registerPassword.addEventListener('input', (e) => {
            checkPasswordStrength(e.target.value);
        });
    }
});

// Exponer función showTab globalmente para uso en HTML
window.showTab = showTab;
