# Habilitar edición automática para el modo Quanty

Propósito
- Guía paso a paso para permitir que el modo `Quanty` edite código y aprenda de errores (bucle de corrección) en este repositorio.

Requisitos previos
- Acceso de administrador o capacidad para configurar el runtime/servicio que carga los chat modes.
- Mecanismo en el runtime para mapear nombres de herramientas (`insert_edit_into_file`, `get_errors`, etc.) a funciones seguras.
- Políticas de auditoría/permiso definidas para evitar cambios no autorizados.

Checklist (lo que haremos con esta guía)
- [ ] Registrar/activar `insert_edit_into_file` en el runtime.
- [ ] Registrar/activar `get_errors` en el runtime.
- [ ] Asegurar permisos y auditoría.
- [ ] Probar con un cambio controlado y validar con `get_errors`.

1) Cambiar el header del chatmode (archivo `Quanty.chatmode.md`)

Actualiza el bloque YAML del archivo `Quanty.chatmode.md` para declarar las herramientas que quieres exponer al agente. Ejemplo:

```yaml
---
description: 'Quanty: agent mode with code-editing and error-driven fixes.'
tools: [insert_edit_into_file, get_errors]
---
```

Nota: ya hay versiones de este archivo en el repo; editar localmente está bien, pero el runtime debe reconocer los nombres.

2) Qué debe implementar el runtime / plataforma

El servicio que carga los chatmodes debe:
- Implementar una función segura `insert_edit_into_file` que acepte parámetros (explicación, filePath, code) y aplique edits atomáticamente o como PRs según política.
- Implementar `get_errors(filePaths: List[str])` que ejecute linters/tests y devuelva diagnósticos.
- Registrar esos endpoints/tools y mapearlos al nombre usado en `tools` del YAML.
- Implementar un control de acceso (solo agentes autorizados o tras aprobación humana).

Ejemplo (esquema de API interna)

- insert_edit_into_file
  - request: { explanation: str, filePath: str, code: str }
  - response: { success: bool, summary: str, diff?: str }

- get_errors
  - request: { filePaths: [str] }
  - response: { errors: [ { file: str, line: int, message: str } ], success: bool }

3) Seguridad y auditoría (recomendado)
- Habilitar logging de cada edición (usuario/agente, timestamp, diff).
- Requerir revisión humana para cambios en archivos críticos (config, infra, secrets).
- Mantener backups/branches automáticas (aplicar cambios en branch y abrir PR).

4) Flujo de prueba controlada (pasos)
A. Habilita las herramientas en el runtime.
B. Pide a Quanty un cambio trivial, por ejemplo:
   "Quanty, añade un comentario `# Prueba de edición Quanty` al principio de `src/cerverus/__init__.py` y luego valida errores."
C. Qué debe hacer el agente (automatizado):
   1. Leer el archivo objetivo.
   2. Aplicar `insert_edit_into_file` con la edición mínima.
   3. Llamar `get_errors` sobre `['src/cerverus/__init__.py']`.
   4. Si `get_errors` reporta errores, analizar y reintentar hasta 3 veces (según la política que definiste en `Quanty.chatmode.md`).
   5. Si errores persisten, reportar resumen y abrir PR/issue con diagnóstico.

Comandos útiles para validación manual local

```bash
# Ejecutar tests (desde la raíz del repo)
pytest -q

# Ejecutar linter/flake (si aplica)
# (ajusta según la configuración del proyecto)
flake8 src || true
```

5) Mensajes/Respuestas que debe devolver Quanty (sugerencia)
- Antes de aplicar cambios: "Voy a aplicar este cambio mínimo: <resumen>. ¿Confirmas?" (si quieres revisión humana)
- Después de aplicar: "Cambio aplicado. Ejecutando validación..."
- Si errores: "Encontré X errores; intento 1/3: [resumen]. Aplicando corrección..."
- Si se agotan intentos: "No pude corregir todos los errores en 3 intentos. Resumen: ..., sugerencias: ..."

6) Qué hacer si sigues viendo "Unknown tool" en los logs
- Significa que el runtime que carga el chatmode no está exponiendo la herramienta con ese nombre.
- Acciones: registrar la implementación en el runtime o hablar con el administrador del servicio para mapear esas funciones.

7) Alternativa segura (si no quieres habilitar ediciones automáticas)
- Mantener `tools: []` en `Quanty.chatmode.md` y usar Quanty para generar patches o PRs. El flujo sería: Quanty genera el diff o el contenido del archivo; un humano aplica el cambio o revisa el PR.

8) Implementación de ejemplo (snippet para equipo de infra)
- En el runner, añadir handler para `insert_edit_into_file` que cree una rama `quanty/auto-edit/{timestamp}`, aplique cambios, corra tests y cree PR automáticamente si todo pasa, o deje comentario y archivo con errores si falla.

9) Próximos pasos que puedo hacer por ti
- Si me confirmas que habilitaste las herramientas: hago una prueba controlada (aplicando un cambio trivial y ejecutando `get_errors`).
- Si prefieres, puedo generar el código del handler de ejemplo (p. ej. un pequeño webhook en Python) para que tu equipo de infra lo despliegue.

---
Archivo creado: `.github/docs/enable-quanty-editing.md`

Si quieres que haga la prueba ahora, confirma que el runtime expone `insert_edit_into_file` y `get_errors`. Si no, dime si quieres que genere el handler de ejemplo para tu equipo de infra.
