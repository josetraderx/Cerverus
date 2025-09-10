@echo off
echo 🔍 MAPEO COMPLETO DEL PROYECTO CERVERUS - DESDE CERO
echo ==================================================
echo Directorio actual: %cd%
echo Fecha: %date% %time%
echo(

echo 1. ESTRUCTURA DE CARPETAS (Primer Nivel):
echo =========================================
dir /b /ad

echo(
echo 2. ESTRUCTURA COMPLETA (Árbol de Directorios):
echo ================================================
echo Usando tree (si está instalado):
if exist "%ProgramFiles%\Windows Kits\10\Tools\x64\x64\tree.com" (
    "%ProgramFiles%\Windows Kits\10\Tools\x64\x64\tree.com" /f /a
) else (
    echo tree no está instalado.  Use un explorador de archivos.
)

echo(
echo 3. ARCHIVOS PYTHON POR CARPETA:
echo =================================
echo(
echo 3.1 Archivos en raíz:
dir /b *.py

echo(
echo 3.2 Archivos en /api:
dir /b /s /a-d api\*.py 2>nul || echo Carpeta /api no existe o sin archivos .py

echo(
echo 3.3 Archivos en /src:
dir /b /s /a-d src\*.py 2>nul || echo Carpeta /src no existe o sin archivos .py

echo(
echo 3.4 Otros archivos Python importantes:
dir /b /s *.py | findstr /v "__pycache__" | findstr /v ".venv" | findstr /n "^" | findstr /r /c:"^[1-30]:" | findstr /r /c:":"

echo(
echo 4. ARCHIVOS __init__.py (Estructura de Módulos):
echo =================================================
dir /b /s __init__.py | findstr /n "^" | findstr /r /c:"^[1-20]:" | findstr /r /c:":"

echo(
echo 5. ARCHIVOS DE CONFIGURACIÓN:
echo =============================
echo 5.1 Docker:
dir /b /s Dockerfile* docker-compose*.yml

echo(
echo 5.2 Python:
dir /b pyproject.toml requirements.txt setup.py poetry.lock 2>nul || echo Archivos de configuración Python no encontrados en raíz

echo(
echo 6. CONTENIDO DE ARCHIVOS PRINCIPALES:
echo ====================================

echo(
echo 6.1 Contenido de docker-compose.yml (si existe):
type docker-compose.yml 2>nul || echo docker-compose.yml no existe

echo(
echo 6.2 Contenido de pyproject.toml (primeras 20 líneas):
type pyproject.toml | findstr /n "^" | findstr /r /c:"^[1-20]:" | findstr /r /c:":"

echo(
echo 7. ANÁLISIS DE IMPORTS ACTUALES:
echo ================================
echo 7.1 Todos los imports que contienen 'cerverus':
findstr /i /c:"import cerverus" *.py /s | findstr /n "^" | findstr /r /c:"^[1-15]:" | findstr /r /c:":"

echo(
echo 7.2 Todos los imports que contienen 'models':
findstr /i /c:"import models" *.py /s | findstr /n "^" | findstr /r /c:"^[1-15]:" | findstr /r /c:":"

echo(
echo 8. ESTRUCTURA ESPECÍFICA DE MODELOS:
echo ===================================
echo 8.1 ¿Existe carpeta models?
dir /b /ad models 2>nul || echo Carpeta models no encontrada

echo(
echo 8.2 Archivos dentro de carpetas models:
dir /b /s /a-d models\*.py | findstr /n "^" | findstr /r /c:"^[1-15]:" | findstr /r /c:":"

echo(
echo 8.3 Nombres de algoritmos/detectores actuales:
dir /b /s *detector*.py *forest*.py *lstm*.py *autoencoder*.py | findstr /n "^" | findstr /r /c:"^[1-10]:" | findstr /r /c:":"

echo(
echo 9. MAIN/ENTRY POINTS:
echo ===================
echo 9.1 Archivos main.py:
dir /b /s main.py

echo(
echo 9.2 Contenido de main.py principal (si existe):
for %%a in (main.py api\app\main.py src\main.py) do (
    if exist "%%a" (
        echo === %%a ===
        type "%%a" | findstr /n "^" | findstr /r /c:"^[1-15]:" | findstr /r /c:":"
        echo(
    )
)

echo(
echo 10. RESUMEN DE ARCHIVOS PROBLEMÁTICOS:
echo ======================================
echo 10.1 anomaly_detection.py (el que falla):
if exist api\app\endpoints\anomaly_detection.py (
    echo === CONTENIDO DE api\app\endpoints\anomaly_detection.py ===
    type api\app\endpoints\anomaly_detection.py | findstr /n "^" | findstr /r /c:"^[1-20]:" | findstr /r /c:":"
) else (
    echo api\app\endpoints\anomaly_detection.py no encontrado
)

echo(
echo 🎯 MAPEO COMPLETO TERMINADO
echo ==========================
echo Con esta información podremos:
echo 1. Entender la estructura real
echo 2. Identificar nombres reales de archivos
echo 3. Corregir imports específicos
echo 4. Crear solución personalizada