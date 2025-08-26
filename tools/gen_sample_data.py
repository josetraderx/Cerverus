"""Generador simple de datos de precios para PoC.

Genera un JSON imprimible con precios normales y algunos outliers inyectados.
"""
import json
import random
from datetime import datetime, timedelta

def generate(symbol='ABC', n=200, n_outliers=3):
    start = datetime.utcnow() - timedelta(days=n)
    rows = []
    for i in range(n):
        ts = (start + timedelta(minutes=30*i)).isoformat()
        price = random.gauss(100, 2)
        volume = int(abs(random.gauss(1000, 100)))
        rows.append({"timestamp": ts, "symbol": symbol, "price": price, "volume": volume})

    # inject outliers
    for i in range(n_outliers):
        idx = random.randint(0, n-1)
        rows[idx]['price'] = rows[idx]['price'] * 10  # big spike

    print(json.dumps({"data": rows}))

if __name__ == '__main__':
    generate()
