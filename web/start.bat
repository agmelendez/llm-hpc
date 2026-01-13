@echo off
REM Script de inicio rápido para Windows

echo ========================================================
echo   Portal LLM-HPC - Script de Inicio (Windows)
echo   Instalacion y Ejecucion Automatica
echo ========================================================
echo.

REM Verificar directorio
if not exist "backend" (
    echo Error: Este script debe ejecutarse desde el directorio web/
    pause
    exit /b 1
)

REM Navegar a backend
cd backend

echo Instalando dependencias de Python...
pip install -r requirements.txt >nul 2>&1

if errorlevel 1 (
    echo Instalacion con advertencias. Intentando continuar...
)

echo Dependencias instaladas
echo.

REM Configurar variables de entorno
set FLASK_ENV=development
set PORT=5000

echo Iniciando servidor Flask...
echo.
echo ========================================================
echo   Servidor ejecutandose en:
echo   http://localhost:5000
echo.
echo   Accede a:
echo   - Login: http://localhost:5000/login.html
echo   - Portal: http://localhost:5000/index.html
echo.
echo   Presiona Ctrl+C para detener el servidor
echo ========================================================
echo.

REM Ejecutar aplicacion
python app.py
