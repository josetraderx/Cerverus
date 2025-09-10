---
title: "Reglas Globales de Comunicación"
version: "1.0"
owner: "Equipo de Desarrollo"
contact: "#team-dev-standards"
last_updated: "2025-09-09"
---

# Reglas Globales de Comunicación

## Checklist de Calidad para Comunicación
- [ ] Propósito del documento claramente definido
- [ ] Ejemplos de código incluidos y testeados
- [ ] Estándares de comentarios aplicados consistentemente
- [ ] Guidelines de PR/Code review definidas
- [ ] Canal de comunicación para dudas especificado

## Tono y Estilo

### Principio de Comunicación Profesional
- Mantener tono técnico preciso pero accesible
- Usar terminología exacta sin ser excesivamente verboso
- Priorizar claridad sobre complejidad innecesaria
- Evitar jerga innecesaria que no añada valor técnico

### Estilo de Documentación
- Usar formato Markdown consistente con headers jerárquicos
- Incluir ejemplos de código cuando sea relevante
- Mantener párrafos concisos (máximo 4 líneas)
- Usar listas numeradas para procesos secuenciales
- Usar listas con viñetas para elementos no secuenciales

### Idioma y Localización
- Documentación técnica principal en español
- Código y comentarios en inglés
- Variables y funciones en inglés con nomenclatura clara
- Documentación de APIs en inglés para compatibilidad internacional

## Comunicación de Errores y Problemas

### Reporte de Errores
- Incluir siempre: contexto, pasos para reproducir, resultado esperado vs actual
- Proporcionar logs relevantes con timestamps
- Especificar versiones de dependencias y entorno
- Incluir nivel de severidad: crítico, alto, medio, bajo

### Comunicación de Soluciones
- Explicar el problema raíz antes de la solución
- Incluir alternativas consideradas y por qué se descartaron
- Proporcionar estimaciones de tiempo y recursos
- Documentar posibles efectos secundarios o riesgos

## Estándares de Comentarios en Código

### Comentarios en Funciones
```python
def calculate_fraud_score(transaction_data: dict) -> float:
    """
    Calcula score de fraude usando ensemble de algoritmos.
    
    Args:
        transaction_data: Diccionario con datos de transacción
            - amount: Monto de la transacción
            - timestamp: Timestamp ISO 8601
            - merchant: Identificador del comerciante
    
    Returns:
        Score de fraude entre 0.0 y 1.0 (1.0 = alta probabilidad de fraude)
    
    Raises:
        ValueError: Si transaction_data no contiene campos requeridos
        ModelNotLoadedError: Si modelos ML no están inicializados
    """
```

### Comentarios en Configuración
```yaml
# Configuración para detección de fraude en tiempo real
# Ajustar thresholds basado en tolerancia al riesgo del negocio
fraud_detection:
  # Score mínimo para generar alerta (0.0-1.0)
  alert_threshold: 0.7
  
  # Score mínimo para bloqueo automático (0.0-1.0)  
  block_threshold: 0.9
```

## Comunicación en Revisiones de Código

### Feedback Constructivo
- Enfocarse en el código, no en la persona
- Proporcionar sugerencias específicas, no solo señalar problemas
- Explicar el "por qué" detrás de cada comentario
- Reconocer implementaciones bien hechas

### Estructura de Comentarios en PR
```markdown
## Cambios Principales
- [Descripción concisa de cambios]

## Impacto en el Sistema  
- [Áreas afectadas y posibles efectos]

## Testing Realizado
- [Pruebas ejecutadas y resultados]

## Consideraciones de Despliegue
- [Pasos especiales o precauciones necesarias]
```