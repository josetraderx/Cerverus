#!/bin/bash
# verify_env.sh

echo "=== VERIFICANDO ENTORNO PARA ISOLATION FOREST ==="

# Verificar Python
echo "1. Verificando Python..."
python --version

# Verificar librerías críticas
echo "2. Verificando librerías..."
python -c "
import pandas as pd
print(f'✅ pandas: {pd.__version__}')

import numpy as np  
print(f'✅ numpy: {np.__version__}')

try:
    from sklearn.ensemble import IsolationForest
    import sklearn
    print(f'✅ scikit-learn: {sklearn.__version__}')
except ImportError:
    print('❌ scikit-learn no encontrado - INSTALAR: pip install scikit-learn')

try:
    import psycopg2
    print(f'✅ psycopg2: {psycopg2.__version__}')
except ImportError:
    print('❌ psycopg2 no encontrado - INSTALAR: pip install psycopg2-binary')

try:
    import joblib
    print(f'✅ joblib: {joblib.__version__}')
except ImportError:
    print('❌ joblib no encontrado - INSTALAR: pip install joblib')

try:
    import matplotlib.pyplot as plt
    import matplotlib
    print(f'✅ matplotlib: {matplotlib.__version__}')
except ImportError:
    print('❌ matplotlib no encontrado - INSTALAR: pip install matplotlib')

try:
    import seaborn as sns
    print(f'✅ seaborn: {sns.__version__}')
except ImportError:
    print('❌ seaborn no encontrado - INSTALAR: pip install seaborn')
"

echo "3. Verificando conexión a base de datos..."
python <<'EOF'
import psycopg2
try:
    conn = psycopg2.connect(
        host='localhost',
        database='cerverus_db', 
        user='joseadmin',
        password='Jireh2023.',
        port=5433  # Puerto actualizado a 5433
    )
    print('✅ Conexión a PostgreSQL exitosa')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM stock_prices;')
    count = cursor.fetchone()[0]
    print(f'✅ Registros en stock_prices: {count}')
    
    cursor.execute('SELECT COUNT(DISTINCT symbol) FROM stock_prices;')
    symbols = cursor.fetchone()[0]
    print(f'✅ Símbolos únicos: {symbols}')
    
    conn.close()
except Exception as e:
    print(f'❌ Error de conexión: {e}')
EOF

echo "=== VERIFICACIÓN COMPLETA ==="
