"""
Tests básicos de componentes de seguridad
Ejecutar con: python test_security.py
"""

from security import (
    SecurityHeaders,
    InputValidator,
    CSRFProtection,
    PasswordPolicy,
    RateLimitTracker
)

def test_input_validation():
    """Prueba validación de entradas"""
    print("🧪 Testing Input Validation...")

    # Username
    assert InputValidator.validate_username("user123"), "Username válido falló"
    assert not InputValidator.validate_username("us"), "Username corto no detectado"
    assert not InputValidator.validate_username("user<script>"), "Username con script no detectado"

    # Email
    assert InputValidator.validate_email("test@example.com"), "Email válido falló"
    assert not InputValidator.validate_email("invalid-email"), "Email inválido no detectado"

    # Password
    assert InputValidator.validate_password("Password123"), "Password válido falló"
    assert not InputValidator.validate_password("weak"), "Password débil no detectado"
    assert not InputValidator.validate_password("nodigits"), "Password sin dígitos no detectado"

    print("  ✅ Input Validation: PASSED")

def test_sanitization():
    """Prueba sanitización de entradas"""
    print("🧪 Testing Input Sanitization...")

    # Sanitización básica
    dangerous = "<script>alert('XSS')</script>"
    safe = InputValidator.sanitize_input(dangerous)
    assert "<script" not in safe.lower(), "Script tag no eliminado"

    # SQL injection patterns
    sql_injection = "'; DROP TABLE users; --"
    assert not InputValidator.validate_sql_injection(sql_injection), "SQL injection no detectado"

    # XSS patterns
    xss = '<img src=x onerror="alert(1)">'
    assert not InputValidator.validate_xss(xss), "XSS no detectado"

    print("  ✅ Input Sanitization: PASSED")

def test_password_policy():
    """Prueba política de contraseñas"""
    print("🧪 Testing Password Policy...")

    # Contraseña fuerte
    strong = PasswordPolicy.check_strength("MyStr0ng!Pass")
    assert strong['strength'] == 'Fuerte', "Contraseña fuerte no reconocida"

    # Contraseña débil
    weak = PasswordPolicy.check_strength("123")
    assert weak['strength'] == 'Débil', "Contraseña débil no reconocida"

    # Contraseñas comunes
    assert PasswordPolicy.is_commonly_used("password"), "Contraseña común no detectada"
    assert PasswordPolicy.is_commonly_used("123456"), "Contraseña común no detectada"

    print("  ✅ Password Policy: PASSED")

def test_csrf_protection():
    """Prueba protección CSRF"""
    print("🧪 Testing CSRF Protection...")

    csrf = CSRFProtection()

    # Generar token
    token = csrf.generate_token()
    assert len(token) > 20, "Token CSRF muy corto"

    # Validar token
    assert csrf.validate_token(token), "Token válido rechazado"
    assert not csrf.validate_token("invalid_token"), "Token inválido aceptado"

    # Invalidar token
    csrf.invalidate_token(token)
    assert not csrf.validate_token(token), "Token invalidado sigue válido"

    print("  ✅ CSRF Protection: PASSED")

def test_rate_limiting():
    """Prueba rate limiting"""
    print("🧪 Testing Rate Limiting...")

    tracker = RateLimitTracker()
    ip = "192.168.1.1"
    endpoint = "/api/login"

    # Primera request: OK
    assert not tracker.is_rate_limited(ip, endpoint, limit=3, window=60), "Primera request bloqueada"

    # Segunda request: OK
    assert not tracker.is_rate_limited(ip, endpoint, limit=3, window=60), "Segunda request bloqueada"

    # Tercera request: OK
    assert not tracker.is_rate_limited(ip, endpoint, limit=3, window=60), "Tercera request bloqueada"

    # Cuarta request: Bloqueada
    assert tracker.is_rate_limited(ip, endpoint, limit=3, window=60), "Rate limit no aplicado"

    print("  ✅ Rate Limiting: PASSED")

def test_security_headers():
    """Prueba headers de seguridad"""
    print("🧪 Testing Security Headers...")

    class MockResponse:
        def __init__(self):
            self.headers = {}

    security = SecurityHeaders()
    response = MockResponse()
    response = security.add_headers(response)

    # Verificar headers críticos
    assert 'Content-Security-Policy' in response.headers, "CSP header faltante"
    assert 'X-Content-Type-Options' in response.headers, "X-Content-Type-Options faltante"
    assert 'X-Frame-Options' in response.headers, "X-Frame-Options faltante"
    assert 'X-XSS-Protection' in response.headers, "X-XSS-Protection faltante"

    print("  ✅ Security Headers: PASSED")

def run_all_tests():
    """Ejecuta todas las pruebas"""
    print("\n" + "="*60)
    print("  🛡️  TESTS DE SEGURIDAD - Portal LLM-HPC")
    print("="*60 + "\n")

    try:
        test_input_validation()
        test_sanitization()
        test_password_policy()
        test_csrf_protection()
        test_rate_limiting()
        test_security_headers()

        print("\n" + "="*60)
        print("  ✅ TODOS LOS TESTS PASARON CORRECTAMENTE")
        print("="*60 + "\n")

        print("📋 Resumen de Seguridad:")
        print("  ✅ Validación de entradas")
        print("  ✅ Sanitización anti-XSS y SQL Injection")
        print("  ✅ Política de contraseñas fuertes")
        print("  ✅ Protección CSRF")
        print("  ✅ Rate limiting")
        print("  ✅ Headers de seguridad HTTP")
        print("\n🎉 El sistema de seguridad está funcionando correctamente!\n")

        return True

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {str(e)}\n")
        return False
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {str(e)}\n")
        return False

if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
