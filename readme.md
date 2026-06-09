# 🐛 Deep Bug - Módulo de Entrenamiento (AI)

Este repositorio contiene los scripts de preparación, arquitectura y entrenamiento del modelo de Inteligencia Artificial (**MobileNetV3**) utilizado por el sistema **Deep Bug** para la identificación asistida de macroinvertebrados y el cálculo del índice BMWP/Mex.

---

## 📊 Dataset de Macroinvertebrados

Por motivos de optimización, rendimiento del control de versiones y respeto a las políticas de almacenamiento de GitHub, las imágenes originales utilizadas para el entrenamiento del modelo convolucional **no se encuentran almacenadas directamente en este repositorio**.

En su lugar, el dataset completo ha sido empaquetado y alojado en un servidor de almacenamiento externo.

* 🔗 **Enlace de Descarga Oficial:** [Descargar Dataset Completo (Google Drive)](#) *[(Pega aquí tu enlace de Drive)](https://drive.google.com/file/d/1tewjcs1tZxOXECB45bGiZZXCfRKAMNUY/view?usp=sharing)*
* 📦 **Formato del archivo:** Archivo Comprimido (`.zip`)
* 🖼️ **Contenido:** Imágenes reales clasificadas por familias taxonómicas para el biomonitoreo de la calidad del agua.

### 🛠️ Instrucciones de Uso Local

Si deseas replicar el entrenamiento o ejecutar el script `entrenar.py` en tu entorno local, sigue estos pasos:

1. **Clona este repositorio** en tu máquina local.
2. **Descarga el archivo ZIP** desde el enlace de Google Drive proporcionado arriba.
3. **Descomprime el archivo** en la raíz de este proyecto.
4. Asegúrate de que la estructura de carpetas quede de la siguiente manera para que el script de Python pueda localizar las imágenes correctamente:

```text
deepbug_training/
│
├── .gitignore
├── entrenar.py
├── README.md
└── [Nombre_de_tu_carpeta_de_imagenes]/
    ├── Familia_1/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── Familia_2/
        ├── img3.jpg
        └── ...