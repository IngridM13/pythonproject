# Infraestructura del Proyecto

Este directorio contiene los archivos de configuración de infraestructura para diferentes entornos del proyecto.

## Estructura

- `docker-compose.yml` - Configuración principal para entorno de desarrollo
- `docker-compose.test.yml` - Configuración para entorno de pruebas automatizadas
- `Dockerfile.test` - Definición de la imagen Docker para ejecutar tests
- `volumes/` - Directorios para persistencia de datos de los servicios

## Entorno de Tests

El archivo `docker-compose.test.yml` configura un entorno aislado para ejecutar todas las pruebas automatizadas del proyecto. Este entorno incluye:

- **Milvus**: Base de datos vectorial
- **Servicios auxiliares**: Etcd y MinIO para Milvus

### Requisitos

- Docker y Docker Compose instalados en el sistema
- Git (para clonar el repositorio)
- Permisos para ejecutar Docker

### Uso del entorno de tests

#### Iniciar el entorno y ejecutar tests desde la raiz del proyecto

`docker-compose -f infra/docker-compose.test.yml up --build`

Este comando:
1. Construye la imagen para tests usando `Dockerfile.test`
3. Ejecuta todos los tests del proyecto
4. Muestra los resultados en la consola

#### Ejecutar tests sin reconstruir imágenes

`bash docker-compose -f infra/docker-compose.test.yml up`

#### Detener el entorno

`bash docker-compose -f infra/docker-compose.test.yml down`

#### Eliminar volúmenes y limpiar completamente

`bash docker-compose -f infra/docker-compose.test.yml down -v`


### Variables de entorno

El entorno de tests utiliza las siguientes variables, configuradas automáticamente:

| Variable | Valor por defecto | Descripción |
|----------|-------------------|-------------|
| MILVUS_URI | http://milvus:19530 | URL de conexión a Milvus |
| LOG_LEVEL | INFO | Nivel de logging |

## Entorno de Desarrollo

Para el entorno de desarrollo, use:

`bash docker-compose -f infra/docker-compose.yml up`

## Solución de problemas

### Errores de permisos en volúmenes

Si encuentras errores de permisos al acceder a los volúmenes:

## bash

Desde la raíz del proyecto 

`sudo chown -R USER:USER infra/volumes/`


### Error de conexión a Milvus

Si los tests no pueden conectarse a Milvus, asegúrese de que:
1. El servicio Milvus esté funcionando correctamente
2. La variable MILVUS_URI esté correctamente configurada
3. No exista un firewall bloqueando la conexión

Para verificar si Milvus está funcionando:

bash docker-compose -f infra/docker-compose.test.yml ps milvus


