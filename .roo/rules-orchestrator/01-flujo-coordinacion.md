# Flujo de Coordinación - Orquestador

## Propósito

Definir normas y un template para la orquestación y coordinación de pipelines (DAGs, tareas, dependencias) que garantice coherencia con el resto de las reglas de `.roo`.

## Alcance

Aplica a todos los DAGs, fábricas de DAGs dinámicos y TaskGroups usados por Airflow u otros orquestadores en el proyecto Cerverus.

## Estructura recomendada del documento

- Título y versión
- Propósito y alcance
- Responsables / Owner
- Requisitos (funcionales y no funcionales)
- Template de DAG / TaskGroup con example mínimo
- Checklist de calidad (tests, observabilidad, seguridad, rollback)

## Template mínimo de flujo (ejemplo)

```markdown
## Ejemplo: DAG de detección de fraude por símbolo

- DAG id: fraud_detection_{symbol}
- Schedule: adaptivo por riesgo (high: 5m, medium: 15m, low: 1h)
- Max active runs: 3 (high) / 1 (otros)

TaskGroups:
- data_extraction: extract_realtime, extract_historical
- fraud_detection: run_statistical, run_ml, run_ensemble
- post_processing: persist_signals, notify_alerts

Dependencias: data_extraction >> fraud_detection >> post_processing
```

## Checklist de calidad (obligatorio para cada regla de orquestación)

- [ ] Header con propósito y fecha/versión
- [ ] Definido owner y canal de contacto
- [ ] Template de DAG o snippet de ejemplo incluido
- [ ] Tests de integración o instrucciones de test disponibles
- [ ] Observabilidad: métricas y logs definidos (trace_id, latencias)
- [ ] Seguridad: permisos, secretos y limits verificados
- [ ] Plan de rollback y backfill documentado

## Owner

- Equipo: Plataforma / Data Engineering
- Contacto: `#team-data-platform` (Slack) / data-platform@example.com

## Notas

Mantener el documento en español; el código de ejemplo y nombres técnicos pueden quedar en inglés según la guía global. Cuando corresponda, referenciar `rules/03-estilo-codigo.md` para convención de nombres y `rules/02-flujo-trabajo.md` para checkpoints.

