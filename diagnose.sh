#!/bin/bash

echo "🔍 MAPEO COMPLETO DEL PROYECTO CERVERUS - DESDE CERO"
echo "===================================================="
echo "Directorio actual: $(pwd)"
echo "Fecha: $(date)"
echo ""

echo "1. ESTRUCTURA DE CARPETAS (Primer Nivel):"
echo "========================================="
ls -la | grep "^d"

echo ""
echo "2. ESTRUCTURA COMPLETA (Árbol de Directorios):"
echo "=============================================="
if command -v tree >/dev/null 2>&1; then
    tree -d -L 4
else
    echo "Usando find (tree no disponible):"
    find . -type d | head -30 | sort
fi

echo ""
echo "3. ARCHIVOS PYTHON POR CARPETA:"
echo "==============================="
echo ""
echo "3.1 Archivos en raíz:"
find . -maxdepth 1 -name "*.py" -type f

echo ""
echo "3.2 Archivos en /api:"
find ./api -name "*.py" -type f 2>/dev/null | head -20 || echo "Carpeta /api no existe o sin archivos .py"

echo ""
echo "3.3 Archivos en /src:"
find ./src -name "*.py" -type f 2>/dev/null | head -20 || echo "Carpeta /src no existe o sin archivos .py"

echo ""
echo "3.4 Otros archivos Python importantes:"
find . -name "*.py" -type f | grep -v "__pycache__" | grep -v ".venv" | head -30

echo ""
echo "4. ARCHIVOS __init__.py (Estructura de Módulos):"
echo "==============================================="
find . -name "__init__.py" | head -20

echo ""
echo "5. ARCHIVOS DE CONFIGURACIÓN:"
echo "============================="
echo "5.1 Docker:"
find . -name "Dockerfile*" -o -name "docker-compose*.yml"

echo ""
echo "5.2 Python:"
ls -la pyproject.toml requirements.txt setup.py poetry.lock 2>/dev/null || echo "Archivos de configuración Python no encontrados en raíz"

echo ""
echo "6. CONTENIDO DE ARCHIVOS PRINCIPALES:"
echo "===================================="

echo ""
echo "6.1 Contenido de docker-compose.yml (si existe):"
if [ -f "docker-compose.yml" ]; then
    cat docker-compose.yml
else
    echo "docker-compose.yml no existe"
fi

echo ""
echo "6.2 Contenido de pyproject.toml (primeras 20 líneas):"
if [ -f "pyproject.toml" ]; then
    head -20 pyproject.toml
else
    echo "pyproject.toml no existe"
fi

echo ""
echo "7. ANÁLISIS DE IMPORTS ACTUALES:"
echo "==============================="
echo "7.1 Todos los imports que contienen 'cerverus':"
grep -r "import.*cerverus\|from.*cerverus" . --include="*.py" | head -15

echo ""
echo "7.2 Todos los imports que contienen 'models':"
grep -r "import.*models\|from.*models" . --include="*.py" | head -15

echo ""
echo "8. ESTRUCTURA ESPECÍFICA DE MODELOS:"
echo "==================================="
echo "8.1 ¿Existe carpeta models?"
find . -name "models" -type d

echo ""
echo "8.2 Archivos dentro de carpetas models:"
find . -path "*/models/*" -name "*.py" | head -15

echo ""
echo "8.3 Nombres de algoritmos/detectores actuales:"
find . -name "*detector*.py" -o -name "*forest*.py" -o -name "*lstm*.py" -o -name "*autoencoder*.py" | head -10

echo ""
echo "9. MAIN/ENTRY POINTS:"
echo "===================="
echo "9.1 Archivos main.py:"
find . -name "main.py"

echo ""
echo "9.2 Contenido de main.py principal (si existe):"
for file in main.py api/app/main.py src/main.py; do
    if [ -f "$file" ]; then
        echo "=== $file ==="
        head -15 "$file"
        echo ""
    fi
done

echo ""
echo "10. RESUMEN DE ARCHIVOS PROBLEMÁTICOS:"
echo "====================================="
echo "10.1 anomaly_detection.py (el que falla):"
if [ -f "api/app/endpoints/anomaly_detection.py" ]; then
    echo "=== CONTENIDO DE api/app/endpoints/anomaly_detection.py ==="
    head -20 "api/app/endpoints/anomaly_detection.py"
else
    echo "api/app/endpoints/anomaly_detection.py no encontrado"
fi

echo ""
echo "🎯 MAPEO COMPLETO TERMINADO"
echo "=========================="
echo "Con esta información podremos:"
echo "1. Entender la estructura real"
echo "2. Identificar nombres reales de archivos"
echo "3. Corregir imports específicos"
echo "4. Crear solución personalizada"