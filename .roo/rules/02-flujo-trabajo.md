---
title: "Reglas Globales de Flujo de Trabajo"
version: "1.0"
owner: "Equipo de Desarrollo"
contact: "#team-dev-standards"
last_updated: "2025-09-09"
---

# Reglas Globales de Flujo de Trabajo

## Checklist de Calidad para Flujo de Trabajo
- [ ] Metodología Pensar-Hacer-Revisar aplicada
- [ ] Checkpoints obligatorios completados
- [ ] Documentación concurrente realizada
- [ ] Testing continuo implementado
- [ ] Plan de rollback definido cuando aplique

## Metodología Pensar-Hacer-Revisar

### Fase 1: PENSAR
Antes de cualquier implementación o cambio significativo:

1. **Análisis del Problema**
   - Definir claramente el problema a resolver
   - Identificar restricciones y limitaciones
   - Entender el contexto y dependencias

2. **Exploración de Soluciones**
   - Investigar al menos 3 enfoques diferentes
   - Evaluar pros y contras de cada opción
   - Considerar impacto en performance, mantenibilidad y escalabilidad

3. **Planificación Detallada**
   - Crear plan de implementación paso a paso
   - Identificar riesgos y estrategias de mitigación
   - Estimar tiempo y recursos necesarios

### Fase 2: HACER
Durante la implementación:

1. **Implementación Incremental**
   - Dividir trabajo en chunks pequeños y testeable
   - Implementar funcionalidad mínima viable primero
   - Validar cada incremento antes de continuar

2. **Documentación Concurrente**
   - Escribir documentación mientras se desarrolla
   - Incluir comentarios explicativos en código complejo
   - Mantener README y changelogs actualizados

3. **Testing Continuo**
   - Escribir tests antes o durante desarrollo
   - Ejecutar tests automatizados frecuentemente
   - Realizar testing manual de casos edge

### Fase 3: REVISAR
Después de completar implementación:

1. **Revisión Técnica**
   - Code review por al menos un compañero
   - Verificar cumplimiento de estándares
   - Validar performance y seguridad

2. **Testing Integral**
   - Ejecutar suite completa de tests
   - Realizar testing de integración
   - Validar en entorno staging similar a producción

3. **Documentación Final**
   - Actualizar documentación técnica
   - Crear/actualizar runbooks operacionales
   - Documentar decisiones de diseño

## Checkpoints Obligatorios

### Checkpoint 1: Definición Clara
**Cuándo**: Antes de comenzar cualquier tarea técnica
**Criterios de Salida**:
- [ ] Problema definido en 1-2 oraciones claras
- [ ] Criterios de éxito específicos y medibles
- [ ] Restricciones y limitaciones identificadas
- [ ] Stakeholders relevantes informados

### Checkpoint 2: Plan de Implementación
**Cuándo**: Antes de escribir código
**Criterios de Salida**:
- [ ] Enfoque técnico seleccionado y justificado
- [ ] Plan de implementación con pasos específicos
- [ ] Estrategia de testing definida
- [ ] Riesgos identificados con planes de mitigación

### Checkpoint 3: Implementación Funcional
**Cuándo**: Después de implementar funcionalidad básica
**Criterios de Salida**:
- [ ] Funcionalidad principal implementada y funcional
- [ ] Tests básicos escritos y pasando
- [ ] Código revisado por al menos una persona
- [ ] Documentación básica completada

### Checkpoint 4: Validación Completa
**Cuándo**: Antes de merge/deploy
**Criterios de Salida**:
- [ ] Todos los tests automatizados pasando
- [ ] Testing manual completado
- [ ] Code review aprobado
- [ ] Documentación actualizada
- [ ] Plan de rollback definido (si aplicable)

## Gestión de Interrupciones

### Principio de Contexto Protegido
- Completar checkpoint actual antes de cambiar de tarea
- Documentar estado actual si interrupción es urgente
- Estimar tiempo real de interrupción antes de aceptar

### Gestión de Context Switching
1. **Antes de Cambiar de Tarea**
   - Documentar estado actual en 2-3 bullets
   - Commitear work in progress si es seguro
   - Estimar tiempo de retorno a tarea original

2. **Al Retomar Tarea**
   - Revisar documentación de estado
   - Re-ejecutar tests para validar estado
   - Continuar desde último checkpoint completado

## Escalación y Toma de Decisiones

### Niveles de Escalación

**Nivel 1 - Decisión Individual** (< 2 horas impacto)
- Implementación de funcionalidad menor
- Refactoring local sin cambios de API
- Fixes de bugs no críticos

**Nivel 2 - Consulta con Equipo** (< 1 día impacto)
- Cambios de API o interfaces
- Nuevas dependencias
- Cambios de arquitectura menores

**Nivel 3 - Revisión Arquitectural** (> 1 día impacto)
- Cambios de arquitectura mayores
- Nuevas tecnologías o frameworks
- Decisiones que afectan múltiples equipos

### Proceso de Escalación
1. **Documentar Contexto**
   - Problema específico y contexto
   - Opciones consideradas
   - Recomendación con justificación

2. **Consultar Nivel Apropiado**
   - Presentar opciones, no solo problemas
   - Incluir impacto en tiempo y recursos
   - Solicitar decisión con timeline específico

3. **Documentar Decisión**
   - Registrar decisión tomada y rationale
   - Comunicar a stakeholders afectados
   - Actualizar documentación relevante