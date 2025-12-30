## Comentarios generales ##

Comentarios Bernie:
- Cambié la estructura del proyecto, lo acomodé en paquetes y demás. Revisate eso. 
- Anoté algunas cosas en el mismo código. 
- Por alguna razón no me corre, no se si es algo de mi mac u otra cosa (otros proy probé y me corren)
- Tengamos una clase / modulo HDC como el que deje en hypervectors. 
- Hagamos un flujo entero: 
  - Generamos datos de prueba
  - Guardamos en postgresql 
  - Generamos el vector de cada fila (teniendo en cuenta los tipos de datos de las columnas)
    - Probamos Milvus y lo indexamos ahí, en vez de pgvector.
  - Revisemos y probemos búsquedas y probamos cambios en las medidas de distancia.
  - Luego generamos, datos de prueba para conciliar. 
  - Corremos una conciliación, medimos tiempos, efectividad, cobertura.


Para el documento: 

- Por qué elegimos tal encoding de vector (binario / bipolar)
- Por qué elegimos el encoding tal o cual dependiendo de los tipo de datos -> apoyemonos en los papers que hablan de esto principalmente. 
- Cómo usamos Milvus y qué tenemos que tener en cuenta para el cálculo de distancias, encoding. 
  - Se extiende de alguna forma el framework implementando una nueva función de embedding, que será el hipervector.

## Setup

El proyecto utiliza `pyenv` para gestionar la versión de Python. Para configurar el entorno de desarrollo, seguir estos pasos:

1.  **Instalar la versión de Python correcta:**
    ```bash
    pyenv install
    ```

2.  **Instalar las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Levantar los servicios de infraestructura:**
    El proyecto utiliza Docker para gestionar los servicios de base de datos. Para levantarlos, ejecutar:
    ```bash
    docker-compose -f infra/docker-compose.yml up -d
    ```

## Testing

Para ejecutar los tests, utilizar el siguiente comando:

```bash
    docker-compose -f infra/docker-compose.test.yml up --build
   ```


o bien dar play a los respectivos archivos.