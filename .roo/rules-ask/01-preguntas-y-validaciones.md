---
title: "Preguntas y Validaciones para Cerverus"
version: "1.0"
owner: "Equipo de Desarrollo"
contact: "#team-dev-standards"
last_updated: "2025-09-09"
---

# Preguntas y Validaciones para Cerverus

## Checklist de Calidad para Preguntas y Validaciones
- [ ] Framework de clarificación estructurada aplicado
- [ ] Preguntas específicas por tipo de tarea definidas
- [ ] Matriz de decisión para casos ambiguos implementada
- [ ] Template de clarificación utilizado
- [ ] Validación de contexto Cerverus realizada

## Framework de Clarificación Estructurada

### Preguntas por Tipo de Tarea

#### Implementación de Algoritmos ML
**Antes de implementar cualquier algoritmo:**
- ¿Qué tier del sistema Cerverus corresponde? (Tier 1: Estadístico, Tier 2: ML No supervisado, Tier 3: Deep Learning, Tier 4: Ensemble)
- ¿Qué features específicas requiere de nuestro feature store?
- ¿Cuál es la latencia máxima aceptable? (<100ms para tiempo real, <5s para batch)
- ¿Cómo se integrará con el ensemble final?
- ¿Qué métricas específicas de fraude debemos optimizar? (Precision vs Recall)

#### Modificaciones de Pipeline de Datos
**Para cambios en las 7 etapas:**
- ¿Qué etapas específicas se ven afectadas? (1-Recolección, 2-Almacenamiento, 3-Procesamiento, 4-Orquestación, 5-ML/Calidad, 6-Infraestructura, 7-Monitoreo)
- ¿El cambio requiere migración de datos existentes?
- ¿Afecta el compliance con PCI-DSS, GDPR, SEC o FINRA?
- ¿Hay dependencias con DAGs de Airflow existentes?
- ¿Requiere actualización de esquemas en dbt?

#### Features de API y Integración
**Para nuevas APIs o modificaciones:**
- ¿Es para tiempo real (<100ms) o batch?
- ¿Qué nivel de autenticación requiere? (Basic, JWT, mTLS)
- ¿Necesita rate limiting específico por cliente?
- ¿Debe cumplir estándares específicos de compliance?
- ¿Cómo se versionará la API?

### Validación de Requerimientos

#### Checklist de Validación Técnica
```markdown
## Validación Técnica Pre-Implementación

### Performance y Escalabilidad
- [ ] ¿Cumple con SLA de latencia definido?
- [ ] ¿Soporta el throughput requerido (10K transacciones/segundo)?
- [ ] ¿Es horizontalmente escalable?
- [ ] ¿Tiene plan de auto-scaling definido?

### Seguridad y Compliance
- [ ] ¿Cumple con normativas financieras aplicables?
- [ ] ¿Maneja datos sensibles correctamente?
- [ ] ¿Tiene auditoría de accesos implementada?
- [ ] ¿Implementa principio de menor privilegio?

### Observabilidad
- [ ] ¿Tiene métricas de negocio y técnicas?
- [ ] ¿Implementa logging estructurado?
- [ ] ¿Está integrado con tracing distribuido?
- [ ] ¿Tiene alertas críticas definidas?

### Integración
- [ ] ¿Se integra correctamente con el ensemble ML?
- [ ] ¿Es compatible con arquitectura de microservicios?
- [ ] ¿Tiene manejo de circuit breaker?
- [ ] ¿Implementa retry con backoff exponencial?
```

#### Preguntas de Negocio Críticas
1. **Impacto en ROI**: ¿Cómo contribuye a los $2M+ anuales objetivo?
2. **Reducción de Falsos Positivos**: ¿Ayuda a mantener <10% false positive rate?
3. **Compliance**: ¿Afecta reportes regulatorios automáticos?
4. **Experiencia de Usuario**: ¿Mantiene tiempo de investigación <15 minutos?

### Proceso de Validación por Etapa

#### Etapa 1-2: Datos (Recolección + Almacenamiento)
```markdown
### Preguntas Específicas para Datos
- ¿Qué fuente específica? (Yahoo Finance, SEC EDGAR, FINRA, Alpha Vantage)
- ¿Frecuencia de actualización requerida?
- ¿Formato y esquema de datos esperado?
- ¿Estrategia de reconciliación entre fuentes?
- ¿Política de retención específica?
- ¿Particionamiento en S3 requerido?
- ¿Nivel de transformación? (Bronze crudo, Silver curado, Gold analytics-ready)
```

#### Etapa 3-4: Procesamiento + Orquestación
```markdown
### Preguntas para Pipeline Processing
- ¿Es procesamiento batch o streaming?
- ¿Qué transformaciones dbt específicas?
- ¿Integración con Apache Flink requerida?
- ¿Dependencias con otros DAGs?
- ¿Estrategia de backfill si falla?
- ¿Métricas de calidad de datos a validar?
```

#### Etapa 5: ML y Calidad
```markdown
### Preguntas para Machine Learning
- ¿Algoritmo supervisado o no supervisado?
- ¿Features existentes o nuevas features requeridas?
- ¿Estrategia de entrenamiento (online/offline)?
- ¿Cómo medir drift del modelo?
- ¿Umbral de confianza para alertas?
- ¿Integración con MLflow registry?
```

## Validación de Contexto Cerverus

### Matriz de Decisión para Casos Ambiguos
```python
class CerverusDecisionMatrix:
    """Matriz para validar decisiones técnicas en contexto Cerverus"""
    
    DECISION_CRITERIA = {
        'latency_critical': {
            'question': '¿La funcionalidad debe responder en <100ms?',
            'if_yes': 'Usar streaming pipeline con Apache Flink',
            'if_no': 'Puede usar procesamiento batch con dbt'
        },
        'fraud_detection_impact': {
            'question': '¿Afecta directamente la detección de fraude?',
            'if_yes': 'Requiere validación con equipo ML y testing A/B',
            'if_no': 'Puede seguir proceso de desarrollo estándar'
        },
        'compliance_sensitive': {
            'question': '¿Maneja datos de tarjetas o información personal?',
            'if_yes': 'Requiere revisión de compliance (PCI-DSS/GDPR)',
            'if_no': 'Seguir políticas estándar de seguridad'
        },
        'multi_stage_impact': {
            'question': '¿Afecta múltiples etapas del sistema?',
            'if_yes': 'Requiere coordinación con arquitecto y plan de migración',
            'if_no': 'Puede implementarse de forma aislada'
        }
    }
```

### Template de Clarificación
```markdown
## Template de Clarificación para Cerverus

### Contexto del Request
- **Etapa(s) afectada(s)**: [1-7]
- **Tipo de cambio**: [Feature nueva / Bug fix / Optimización / Refactor]
- **Urgencia**: [Crítico / Alto / Medio / Bajo]
- **Impacto estimado**: [Alto / Medio / Bajo]

### Clarificaciones Técnicas Necesarias
1. **Requisitos Funcionales**:
   - ¿Qué problema específico resuelve?
   - ¿Cuáles son los criterios de éxito medibles?
   - ¿Hay restricciones técnicas específicas?

2. **Requisitos No Funcionales**:
   - ¿Latencia máxima aceptable?
   - ¿Throughput requerido?
   - ¿Disponibilidad necesaria?
   - ¿Requisitos de seguridad específicos?

3. **Integración**:
   - ¿Con qué componentes debe integrarse?
   - ¿Rompe compatibilidad hacia atrás?
   - ¿Requiere cambios en APIs existentes?

4. **Testing y Validación**:
   - ¿Cómo se va a probar?
   - ¿Métricas específicas a monitorear?
   - ¿Estrategia de rollback si falla?

### Ejemplo de Uso
**Request**: "Necesito mejorar la detección de fraude"

**Clarificaciones requeridas**:
- ¿Qué específicamente no está funcionando en la detección actual?
- ¿Te refieres a mejorar precision, recall, o reducir latencia?
- ¿Para qué tier de algoritmos? (1-Estadístico, 2-ML, 3-DL, 4-Ensemble)
- ¿Tienes métricas específicas del problema actual?
- ¿Hay nuevos tipos de fraude que no detectamos?
- ¿Es para tiempo real o análisis batch?
```