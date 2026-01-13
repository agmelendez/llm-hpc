#!/bin/bash

# Script de inicio rápido para el Portal LLM-HPC
# Instala dependencias y ejecuta el servidor

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Portal LLM-HPC - Script de Inicio                        ║"
echo "║  Instalación y Ejecución Automática                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Verificar que estamos en el directorio correcto
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Este script debe ejecutarse desde el directorio web/"
    exit 1
fi

# Navegar a backend
cd backend

echo "📦 Instalando dependencias de Python..."
pip install -r requirements.txt > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "⚠️  Instalación con advertencias. Intentando continuar..."
fi

echo "✅ Dependencias instaladas"
echo ""

# Configurar variables de entorno para desarrollo
export FLASK_ENV=development
export PORT=5000

echo "🚀 Iniciando servidor Flask..."
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Servidor ejecutándose en:                                 ║"
echo "║  http://localhost:5000                                     ║"
echo "║                                                            ║"
echo "║  Accede a:                                                 ║"
echo "║  - Login: http://localhost:5000/login.html                 ║"
echo "║  - Portal: http://localhost:5000/index.html                ║"
echo "║                                                            ║"
echo "║  Presiona Ctrl+C para detener el servidor                 ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Ejecutar aplicación
python app.py
