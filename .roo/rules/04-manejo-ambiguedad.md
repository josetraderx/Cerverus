---
title: "Reglas Globales de Manejo de Ambigüedad"
version: "1.0"
owner: "Equipo de Desarrollo"
contact: "#team-dev-standards"
last_updated: "2025-09-09"
---

# Reglas Globales de Manejo de Ambigüedad

## Checklist de Calidad para Manejo de Ambigüedad
- [ ] Framework de preguntas estructuradas aplicado
- [ ] Validación de entendimiento realizada
- [ ] Documentación de decisiones mantenida
- [ ] Asunciones claramente identificadas y validadas
- [ ] Plan de mitigación de riesgos definido

## Estrategias de Clarificación

### Principio de Clarificación Proactiva
Cuando encuentres ambigüedad o información incompleta, siempre hacer preguntas específicas antes de asumir o proceder con implementación.

### Framework de Preguntas Estructuradas

#### 1. Preguntas de Contexto
```markdown
## Contexto del Problema
- ¿Cuál es el problema específico que estamos resolviendo?
- ¿Quién es el usuario final de esta funcionalidad?
- ¿Cómo se relaciona esto con el objetivo general del proyecto?
- ¿Existen restricciones técnicas o de negocio que deba considerar?
```

#### 2. Preguntas de Alcance
```markdown
## Definición de Alcance
- ¿Qué casos de uso específicos debemos cubrir?
- ¿Qué casos de uso están explícitamente fuera del alcance?
- ¿Cuáles son los criterios de éxito medibles?
- ¿Existen casos edge que debamos considerar prioritarios?
```

#### 3. Preguntas Técnicas
```markdown
## Especificaciones Técnicas
- ¿Hay preferencias de tecnología o arquitectura?
- ¿Existen APIs o sistemas existentes que debamos integrar?
- ¿Cuáles son los requisitos de performance/latencia?
- ¿Hay estándares de seguridad específicos que seguir?
```

#### 4. Preguntas de Implementación
```markdown
## Detalles de Implementación
- ¿Cómo debería manejar errores o casos de fallo?
- ¿Qué nivel de logging/monitoreo necesitamos?
- ¿Hay patrones de código existentes que debamos seguir?
- ¿Cómo debería ser la estrategia de testing?
```

## Técnicas de Clarificación por Tipo

### Ambigüedad en Requerimientos
**Situación**: "Necesitamos detectar transacciones sospechosas"

**Preguntas de Clarificación**:
- ¿Qué define específicamente una transacción como "sospechosa"?
- ¿Hay tipos específicos de patrones que debemos detectar (monto inusual, frecuencia, ubicación geográfica)?
- ¿Cuál es el balance deseado entre falsos positivos y falsos negativos?
- ¿Hay regulaciones específicas que debemos cumplir en la detección?

### Ambigüedad en Especificaciones Técnicas
**Situación**: "El sistema debe ser rápido y escalable"

**Preguntas de Clarificación**:
- ¿Cuál es la latencia máxima aceptable? (ej: <100ms, <1s, <5s)
- ¿Cuántas transacciones por segundo necesitamos soportar actualmente y en el futuro?
- ¿Hay picos específicos de carga que debamos planificar?
- ¿Qué recursos (CPU, memoria, almacenamiento) tenemos disponibles?

### Ambigüedad en Integración
**Situación**: "Integrar con el sistema de pagos existente"

**Preguntas de Clarificación**:
- ¿Qué sistema de pagos específico? ¿Hay documentación de API disponible?
- ¿Es integración en tiempo real o batch?
- ¿Qué datos necesitamos enviar/recibir específicamente?
- ¿Hay autenticación o autorización especial requerida?
- ¿Existe un entorno de testing/sandbox disponible?

## Proceso de Validación de Entendimiento

### Técnica de Resumen Confirmatorio
Después de recibir clarificaciones, resumir entendimiento:

```markdown
## Confirmación de Entendimiento

Basado en nuestra conversación, entiendo que necesito:

### Objetivo Principal
[Descripción clara del objetivo en 1-2 oraciones]

### Funcionalidad Específica
- [Punto específico 1]
- [Punto específico 2]
- [Punto específico 3]

### Restricciones y Limitaciones
- [Restricción técnica 1]
- [Restricción de negocio 1]
- [Limitación de tiempo/recursos]

### Criterios de Éxito
- [Métrica medible 1]
- [Métrica medible 2]

¿Es correcto este entendimiento? ¿Hay algo que deba ajustar o añadir?
```

### Validación Iterativa
Para proyectos complejos, usar validación en etapas:

1. **Validación de Concepto**: Confirmar entendimiento del problema
2. **Validación de Diseño**: Confirmar enfoque técnico propuesto  
3. **Validación de Implementación**: Confirmar detalles específicos
4. **Validación de Criterios**: Confirmar definición de "terminado"

## Manejo de Información Contradictoria

### Escalación de Contradicciones
Cuando encuentres información contradictoria:

1. **Documentar la Contradicción**
   ```markdown
   ## Contradicción Identificada
   
   **Fuente A** dice: [información específica]
   **Fuente B** dice: [información contradictoria]
   
   **Impacto**: [cómo afecta esto la implementación]
   ```

2. **Buscar Clarificación Específica**
   - Preguntar directamente sobre la contradicción
   - Solicitar priorización si ambas son válidas en diferentes contextos
   - Pedir decisión final con justificación

3. **Proponer Solución Temporal**
   - Sugerir implementación que pueda acomodar ambos escenarios
   - Proponer configuración que permita cambiar comportamiento
   - Recomendar enfoque más conservador mientras se resuelve

### Documentación de Decisiones de Ambigüedad
Mantener registro de decisiones tomadas ante ambigüedad:

```markdown
## Registro de Decisión: [Fecha]

### Ambigüedad Original
[Descripción de la ambigüedad o falta de información]

### Opciones Consideradas
1. [Opción 1 con pros/contras]
2. [Opción 2 con pros/contras]
3. [Opción 3 con pros/contras]

### Decisión Tomada
[Opción seleccionada con justificación]

### Asunciones Hechas
- [Asunción 1]
- [Asunción 2]

### Plan de Validación
[Cómo y cuándo validaremos que la decisión fue correcta]
```

## Comunicación de Limitaciones y Riesgos

### Transparencia sobre Incertidumbre
Cuando proceder con información incompleta es necesario:

```markdown
## Proceder con Limitaciones Conocidas

### Lo que Sabemos
- [Información confirmada 1]
- [Información confirmada 2]

### Lo que Asumimos
- [Asunción 1 - riesgo: bajo/medio/alto]
- [Asunción 2 - riesgo: bajo/medio/alto]

### Riesgos Identificados
- [Riesgo 1]: [impacto y probabilidad]
- [Riesgo 2]: [impacto y probabilidad]

### Plan de Mitigación
- [Acción específica para reducir riesgo 1]
- [Checkpoint para validar asunción crítica]
- [Plan B si asunción principal es incorrecta]
```

### Solicitud de Validación Continua
Establecer checkpoints específicos para validar asunciones:

- **Checkpoint temprano**: Validar dirección general después de 25% del trabajo
- **Checkpoint medio**: Validar implementación específica en 50%
- **Checkpoint final**: Validar criterios de éxito antes de considerar terminado