# Cerverus

Plantilla de repositorio para el proyecto Cerverus.

Estructura creada automáticamente.

![CI](https://github.com/josetraderx/Cerverus/actions/workflows/ci.yml/badge.svg)

## Configuración (nuevo)

Se añadió un directorio `config/` en la raíz para centralizar configuraciones por ambiente, modelos y pipelines.

Estructura creada:

```
config/
├── environments/
│   ├── local.yml
│   ├── staging.yml
│   └── production.yml
├── models/
│   ├── tier1_config.yml
│   ├── tier2_config.yml
│   ├── tier3_config.yml
│   └── tier4_config.yml
└── pipelines/
	├── data_ingestion.yml
	└── ml_training.yml
```

Propósito: centralizar parámetros para facilitar despliegues y pruebas en múltiples ambientes.

## Infrastructure (nuevo)

Se añadió un directorio `infrastructure/` con plantillas de Terraform, manifests de Kubernetes y configuraciones de monitoreo.

Estructura creada (resumen):

```
infrastructure/
├── terraform/
│   ├── environments/
│   │   ├── dev.tf
│   │   ├── staging.tf
│   │   └── prod.tf
│   ├── modules/
│   │   ├── database/
│   │   ├── kafka/
│   │   └── kubernetes/
│   └── main.tf
├── kubernetes/
│   ├── manifests/
│   │   ├── api-deployment.yaml
│   │   ├── ml-workers.yaml
│   │   └── kafka-cluster.yaml
│   └── helm/
│       └── cerverus-chart/
└── monitoring/
    ├── grafana/
    │   └── dashboards/
    ├── prometheus/
    │   └── rules/
    └── alertmanager/
        └── config/
```

Todos los archivos son plantillas iniciales; adáptalos a tu proveedor y pipeline CI/CD.
