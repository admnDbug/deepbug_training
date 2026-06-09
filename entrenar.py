import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime

# fijar PYTHONHASHSEED antes de importar cualquier otra cosa
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import keras
from keras import layers
from keras.applications import ResNet50, MobileNetV3Large
from keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, top_k_accuracy_score
)

import shutil

# Configuración general
IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_CLASES  = 5

# Hiperparámetros de la Fase 1: solo cabeza nueva
FASE1_EPOCHS = 55   #Con pocas épocas no lograba el máximo aprendizaje
FASE1_LR     = 1e-3 #Baja de inicio para evitar inestabilidad con la cabeza sin entrenar

# Hiperparámetros de la Fase 2: fine-tuning, se descongelan capas finales del backbone
FASE2_EPOCHS = 80   # Más épocas para permitir que el modelo se recupere de mínimos locales y aproveche el fine-tuning
FASE2_LR     = 1e-5 # LR super baja para no arruinar los pesos preentrenados durante el fine-tuning
CAPAS_DESCONGELAR = {
    "resnet50":  15,   # No subir más de 20 porque resnet50 tiene solo 50
    "mobilenet": 30,   # MobileNetV3Large tiene aprox 130 capas, así que podemos subir más sin riesgo de olvidar lo aprendido
}

# Rates de dropout por fase
DROPOUT_FASE1 = [0.4, 0.3, 0.2]
DROPOUT_FASE2 = [0.3, 0.2, 0.1] 

CONTEOS_CLASES = {
    "Baetidae":        329,
    "Caenidae":        329,   
    "Heptageniidae":   338,
    "Leptohyphidae":   338,
    "Leptophlebiidae": 296,   # única clase aún desbalanceada
}

# Colocar semilla en los generadores aleatorios
random.seed(SEED)           # Python estándar
np.random.seed(SEED)        # NumPy
tf.random.set_seed(SEED)    # TensorFlow / Keras

def calcular_pesos_clase(clases_ordenadas: list) -> dict:
    conteos = np.array([CONTEOS_CLASES.get(c, 1) for c in clases_ordenadas],
                       dtype=np.float64)
    pesos = conteos.max() / conteos   # inverso proporcional
    pesos = pesos / pesos.max()       # normalizar -> máximo base = 1.0

    BOOST_POR_CLASE = {
        "Leptohyphidae":   1.55,
        "Leptophlebiidae": 1.70,
    }
    for i, cls in enumerate(clases_ordenadas):
        if cls in BOOST_POR_CLASE:
            pesos[i] *= BOOST_POR_CLASE[cls]

    return {i: float(p) for i, p in enumerate(pesos)}


@keras.utils.register_keras_serializable(package="deepbug")
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25,
                 label_smoothing: float = 0.10, **kwargs):
        super().__init__(**kwargs)
        self.gamma           = gamma
        self.alpha           = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        # Aplicar label smoothing: mezcla la etiqueta dura con una distribución uniforme
        num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
        y_true_smooth = (y_true * (1.0 - self.label_smoothing)
                         + self.label_smoothing / num_classes)

        ce     = -y_true_smooth * tf.math.log(y_pred)
        p_t    = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)  # p_t sobre la clase real
        factor = tf.pow(1.0 - p_t, self.gamma)
        loss   = self.alpha * factor * ce
        return tf.reduce_sum(loss, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"gamma": self.gamma, "alpha": self.alpha,
                    "label_smoothing": self.label_smoothing})
        return cfg

class ValScore(keras.callbacks.Callback):
    def __init__(self, peso_acc: float = 0.6, peso_loss: float = 0.4):
        super().__init__()
        self.peso_acc  = peso_acc
        self.peso_loss = peso_loss
        self._loss_max = None

    def on_epoch_end(self, epoch, logs=None):
        acc  = logs.get("val_accuracy", 0.0)
        loss = logs.get("val_loss",     0.0)
        if self._loss_max is None or loss > self._loss_max:
            self._loss_max = loss
        loss_norm = loss / self._loss_max
        score = (self.peso_acc * acc) - (self.peso_loss * loss_norm)
        logs["val_score"] = float(score)


class CosineAnnealingWarmRestarts(keras.callbacks.Callback):
    def __init__(self, lr_max: float, lr_min: float = 1e-6,
                 T_0: int = 15, T_mult: float = 1.5, decay_factor: float = 0.6):
        super().__init__()
        self.lr_max       = lr_max
        self.lr_min       = lr_min
        self.T_0          = T_0
        self.T_mult       = T_mult
        self.decay_factor = decay_factor

    def _compute_lr(self, epoch: int) -> float:
        T_cur   = self.T_0
        elapsed = epoch
        cycle   = 0
        while elapsed >= T_cur:
            elapsed -= T_cur
            T_cur    = int(T_cur * self.T_mult)
            cycle   += 1
        lr_max_cycle = self.lr_max * (self.decay_factor ** cycle)
        lr = (self.lr_min
              + (lr_max_cycle - self.lr_min)
              * 0.5 * (1 + np.cos(np.pi * elapsed / T_cur)))
        return float(np.clip(lr, self.lr_min, self.lr_max))

    def on_epoch_begin(self, epoch, logs=None):
        lr = self._compute_lr(epoch)
        self.model.optimizer.learning_rate.assign(lr)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            logs["learning_rate"] = float(self.model.optimizer.learning_rate)


def construir_datasets(ruta_data, seed):
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical", seed=seed),
        keras.layers.RandomRotation(0.5, seed=seed),
        keras.layers.RandomZoom(height_factor=(-0.05, 0.05), width_factor=(-0.05, 0.05), seed=seed),
        keras.layers.RandomTranslation(0.05, 0.05, seed=seed),
        keras.layers.RandomBrightness(0.1, seed=seed),
        keras.layers.RandomContrast(0.1, seed=seed),
    ], name="augmentation")

    # Hiperpárametros del aumento de datos
    PROB_CROP        = 0.7   # Mucha probabilidad de desencuadrar para evitar sesgos de cámara
    PROB_FONDO       = 0.65  # Reemplazo de fondo con probabilidad alta para forzar a la red a aprender características del insecto, no del fondo
    PROB_CUTOUT      = 0.4   # Parches oscuros aleatorios para simular oclusiones parciales (hojas, suciedad) y forzar a la red a aprender formas globales en lugar de detalles locales
    PROB_JITTER      = 0.3   # Variación global de color para simular diferentes condiciones de iluminación y forzar a la red a aprender características morfológicas en lugar de colores específicos del fondo o del insecto
    PROB_BLUR        = 0.2   # Simula desenfoque de cámara/microscopio

    # MixUp: mezcla pares de imágenes del mismo batch y sus etiquetas.
    # similares (Leptohyphidae y Caenidae, Baetidae y Leptophlebiidae).
    PROB_MIXUP       = 0.40  # probabilidad de aplicar MixUp a un batch

    CUTOUT_MIN_FRAC  = 0.05 # Provoca oclusiones leves hasta moderadas
    CUTOUT_MAX_FRAC  = 0.3  # Dejamos que tape hasta un 30% del insecto en casos extremos

    # Umbral base para la máscara de fondo, se ajusta adaptativamente por imagen
    UMBRAL_FONDO_BASE = 40.0  # umbral mínimo (imágenes muy uniformes)
    UMBRAL_FONDO_MAX  = 90.0  # umbral máximo (imágenes con sombras fuertes)
    JITTER_DELTA     = 0.25   # +-25% de variación en cada canal de color

    def random_crop_augmentation(imagen):
        def aplicar():
            # Agrandamos la imagen un poco
            nuevo_tamano = tf.cast(IMG_SIZE * 1.15, tf.int32)
            imagen_agrandada = tf.image.resize(imagen, [nuevo_tamano, nuevo_tamano])
            
            # Recortamos de vuelta al tamaño original de forma aleatoria
            return tf.image.random_crop(imagen_agrandada, size=[IMG_SIZE, IMG_SIZE, 3])
            
        return tf.cond(tf.random.uniform(()) > PROB_CROP,
                       lambda: imagen,
                       aplicar)

    def random_background_augmentation(imagen):
        def aplicar():
            # Patch de 18×18 px en cada esquina
            TAM_ESQ = 18
            esq_tl = tf.reshape(imagen[:TAM_ESQ,  :TAM_ESQ,  :], [-1, 3])
            esq_tr = tf.reshape(imagen[:TAM_ESQ,  -TAM_ESQ:, :], [-1, 3])
            esq_bl = tf.reshape(imagen[-TAM_ESQ:, :TAM_ESQ,  :], [-1, 3])
            esq_br = tf.reshape(imagen[-TAM_ESQ:, -TAM_ESQ:, :], [-1, 3])

            # Media ponderada: cada esquina pesa lo mismo
            color_fondo_estimado = tf.reduce_mean(
                tf.concat([esq_tl, esq_tr, esq_bl, esq_br], axis=0), axis=0
            ) 
            img_std = tf.math.reduce_std(imagen)           # escalar
            umbral = tf.clip_by_value(
                UMBRAL_FONDO_BASE + img_std * 0.35,
                UMBRAL_FONDO_BASE,
                UMBRAL_FONDO_MAX
            )

            diferencia     = tf.abs(imagen - color_fondo_estimado)          # (H,W,3)
            distancia_l2   = tf.norm(diferencia, axis=-1, keepdims=True)    # (H,W,1)
            # sigmoid centrada en el umbral; k controla la pendiente de transición
            k = 0.12
            mascara_suave  = tf.sigmoid(-(distancia_l2 - umbral) * k)      # (H,W,1)

            tipo_fondo = tf.random.uniform(())

            color_base_a = tf.random.uniform((1, 1, 3), 0.0, 255.0)
            ruido_a      = tf.random.normal(tf.shape(imagen), mean=0.0, stddev=20.0)
            fondo_a      = tf.clip_by_value(color_base_a + ruido_a, 0.0, 255.0)

            color_b1 = tf.random.uniform((1, 1, 3), 0.0, 255.0)
            color_b2 = tf.random.uniform((1, 1, 3), 0.0, 255.0)

            # Rampas horizontales y verticales, combinadas en dirección aleatoria
            t_h = tf.cast(tf.linspace(0.0, 1.0, IMG_SIZE), tf.float32)
            t_v = tf.cast(tf.linspace(0.0, 1.0, IMG_SIZE), tf.float32)
            t_h = tf.reshape(t_h, [IMG_SIZE, 1, 1])   # varía por fila
            t_v = tf.reshape(t_v, [1, IMG_SIZE, 1])   # varía por columna
            w   = tf.random.uniform(())               # peso eje horizontal
            t   = w * t_h + (1.0 - w) * t_v          # (H,W,1)
            fondo_b  = color_b1 * (1.0 - t) + color_b2 * t
            ruido_b  = tf.random.normal(tf.shape(imagen), mean=0.0, stddev=10.0)
            fondo_b  = tf.clip_by_value(fondo_b + ruido_b, 0.0, 255.0)

            color_base_c = tf.random.uniform((1, 1, 3), 0.0, 255.0)
            ruido_c      = tf.random.uniform(tf.shape(imagen), 0.0, 255.0)
            fondo_c      = tf.clip_by_value(color_base_c * 0.65 + ruido_c * 0.35, 0.0, 255.0)

            # Sortear el tipo de fondo
            fondo_nuevo = tf.cond(
                tipo_fondo < 0.40,
                lambda: fondo_a,
                lambda: tf.cond(
                    tipo_fondo < 0.75,
                    lambda: fondo_b,
                    lambda: fondo_c
                )
            )

            # alpha controla cuánto se reemplaza el fondo detectado
            alpha = tf.random.uniform((), 0.75, 1.0)
            imagen_mezclada = (
                imagen     * (1.0 - mascara_suave * alpha)
                + fondo_nuevo * (mascara_suave * alpha)
            )
            return tf.clip_by_value(imagen_mezclada, 0.0, 255.0)

        return tf.cond(tf.random.uniform(()) > PROB_FONDO,
                       lambda: imagen,
                       aplicar)

    def jitter_color_global(imagen):
        def aplicar():
            factores = tf.random.uniform((1, 1, 3), 1.0 - JITTER_DELTA, 1.0 + JITTER_DELTA)
            return tf.clip_by_value(imagen * factores, 0.0, 255.0)

        return tf.cond(tf.random.uniform(()) > PROB_JITTER,
                       lambda: imagen,
                       aplicar)

    def cutout_aleatorio(imagen):
        def aplicar_parche(img):
            h = tf.cast(IMG_SIZE, tf.float32)
            w = tf.cast(IMG_SIZE, tf.float32)
            ph = tf.random.uniform((), CUTOUT_MIN_FRAC, CUTOUT_MAX_FRAC) * h
            pw = tf.random.uniform((), CUTOUT_MIN_FRAC, CUTOUT_MAX_FRAC) * w
            cy = tf.random.uniform((), 0.0, h)
            cx = tf.random.uniform((), 0.0, w)

            y1    = tf.cast(tf.clip_by_value(cy - ph / 2, 0, h), tf.int32)
            y2    = tf.cast(tf.clip_by_value(cy + ph / 2, 0, h), tf.int32)
            x1    = tf.cast(tf.clip_by_value(cx - pw / 2, 0, w), tf.int32)
            x2    = tf.cast(tf.clip_by_value(cx + pw / 2, 0, w), tf.int32)
            alto  = tf.maximum(y2 - y1, 1)
            ancho = tf.maximum(x2 - x1, 1)

            color_parche   = tf.random.uniform((1, 1, 3), 0.0, 255.0)
            parche         = tf.ones((alto, ancho, 3), dtype=tf.float32) * color_parche
            mascara_parche = tf.ones((alto, ancho, 3), dtype=tf.float32)

            pad = [[y1, IMG_SIZE - y2], [x1, IMG_SIZE - x2], [0, 0]]
            parche_completo = tf.pad(parche, pad)
            mascara_completa = tf.pad(mascara_parche, pad)

            return img * (1.0 - mascara_completa) + parche_completo * mascara_completa

        def aplicar_uno_o_dos():
            img_con_uno = aplicar_parche(imagen)
            return tf.cond(tf.random.uniform(()) < 0.5,
                           lambda: aplicar_parche(img_con_uno),
                           lambda: img_con_uno)

        return tf.cond(tf.random.uniform(()) > PROB_CUTOUT,
                       lambda: imagen,
                       aplicar_uno_o_dos)

    def blur_suave_aleatorio(imagen):
        def aplicar():
            kernel_1d = tf.constant([0.0625, 0.25, 0.375, 0.25, 0.0625], dtype=tf.float32)
            kernel_2d = tf.tensordot(kernel_1d, kernel_1d, axes=0)        # (5,5)
            kernel    = tf.reshape(kernel_2d, [5, 5, 1, 1])               # (5,5,1,1)

            img_4d = tf.expand_dims(imagen, axis=0)                       # (1,H,W,3)
            canales = tf.split(img_4d, 3, axis=-1)                        # 3×(1,H,W,1)
            canales_blur = [
                tf.nn.conv2d(c, kernel, strides=[1,1,1,1], padding="SAME")
                for c in canales
            ]
            img_blur = tf.concat(canales_blur, axis=-1)                   # (1,H,W,3)
            return tf.clip_by_value(tf.squeeze(img_blur, axis=0), 0.0, 255.0)

        return tf.cond(tf.random.uniform(()) < PROB_BLUR,
                       aplicar,
                       lambda: imagen)

    def escala_grises_aleatoria(imagen):
        def a_grises():
            gris = tf.image.rgb_to_grayscale(imagen)
            return tf.image.grayscale_to_rgb(gris)
            
        return tf.cond(tf.random.uniform(()) < 0.2, 
                       a_grises,
                       lambda: imagen)

    def rot90_aleatorio(imagen):
        def aplicar():
            k = tf.random.uniform((), minval=1, maxval=4, dtype=tf.int32)
            return tf.image.rot90(imagen, k=k)

        return tf.cond(tf.random.uniform(()) < 0.5, aplicar, lambda: imagen)

    def augmentar_dominio(imagen):
        imagen = tf.cast(imagen, tf.float32)
        imagen = random_crop_augmentation(imagen)       # 1. Desencuadrar
        imagen = rot90_aleatorio(imagen)                # 2. Rotación ortogonal
        imagen = random_background_augmentation(imagen) # 3. Cambiar fondo
        imagen = jitter_color_global(imagen)            # 4. Variar iluminación
        imagen = cutout_aleatorio(imagen)               # 5. Parches oscuros
        imagen = blur_suave_aleatorio(imagen)           # 6. Desenfoque leve
        return imagen

    def mixup_batch(images, labels):
        def aplicar():
            batch_size = tf.shape(images)[0]

            lam = tf.random.uniform(())
            lam = tf.maximum(lam, 1.0 - lam)

            # Permutación aleatoria dentro del batch
            indices         = tf.random.shuffle(tf.range(batch_size))
            images_shuffled = tf.gather(images, indices)
            labels_shuffled = tf.gather(labels, indices)

            mixed_images = lam * images + (1.0 - lam) * images_shuffled
            mixed_labels = lam * labels + (1.0 - lam) * labels_shuffled
            return mixed_images, mixed_labels

        return tf.cond(
            tf.random.uniform(()) < PROB_MIXUP,
            aplicar,
            lambda: (images, labels)
        )

    def crear_dataset(split, preprocesar_fn, entrenar=False):
        ruta_split = ruta_data / split
        ds = keras.utils.image_dataset_from_directory(
            ruta_split,
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=None,
            shuffle=entrenar,
            seed=seed,
            label_mode="categorical",
        )
        clases = ds.class_names
        
        if entrenar:
            ds = ds.map(
                lambda x, y: (escala_grises_aleatoria(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            ds = ds.map(
                lambda x, y: (augmentar_dominio(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            ds = ds.map(
                lambda x, y: (augmentation(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            
        ds = ds.map(
            lambda x, y: (preprocesar_fn(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        
        if entrenar:
            ds = ds.repeat(2)

        ds = ds.batch(BATCH_SIZE)

        # MixUp se aplica en el espacio ya preprocesado, a nivel de batch completo
        if entrenar:
            ds = ds.map(
                mixup_batch,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )

        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds, clases

    return crear_dataset


# Connstrucción de modelos

def construir_modelo(nombre: str, num_clases: int):
    entrada = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    if nombre == "resnet50":
        base = ResNet50(
            weights="imagenet",
            include_top=False,
            input_tensor=entrada
        )
        preprocesar_fn = tf.keras.applications.resnet50.preprocess_input
    elif nombre == "mobilenet":
        base = MobileNetV3Large(
            weights="imagenet",
            include_top=False,
            input_tensor=entrada
        )
        preprocesar_fn = tf.keras.applications.mobilenet_v3.preprocess_input
    else:
        raise ValueError(f"Modelo desconocido: {nombre}")

    base.trainable = False

    reg = keras.regularizers.l2(1e-4)

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(256, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(DROPOUT_FASE1[0], name="drop1")(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(DROPOUT_FASE1[1], name="drop2")(x)

    x = layers.Dropout(DROPOUT_FASE1[2], name="drop3")(x)
    salida = layers.Dense(num_clases, activation="softmax")(x)

    modelo = keras.Model(inputs=entrada, outputs=salida, name=nombre)
    return modelo, base, preprocesar_fn


def subir_dropout(modelo, rates: list) -> None:
    nombres = ["drop1", "drop2", "drop3"]
    for nombre, rate in zip(nombres, rates):
        capa = modelo.get_layer(nombre)
        capa.rate = rate
        print(f"  Dropout ajustado: {nombre} → {rate}")

def descongelar_para_finetune(modelo, base, n_capas: int, lr: float):
    # Activamos la base completa primero para poder modificar sus capas internas
    base.trainable = True
    
    # Congelamos todas las capas excepto las últimas 'n_capas'
    for capa in base.layers[:-n_capas]:
        capa.trainable = False

    # Revisamos las últimas 'n_capas' y volvemos a congelar cualquier capa de BatchNormalization para evitar la amnesia
    for capa in base.layers[-n_capas:]:
        if isinstance(capa, keras.layers.BatchNormalization):
            capa.trainable = False

    # Recompilamos con la tasa de aprendizaje baja y la FocalLoss
    modelo.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
    )
    
    return modelo


# Métricas y visualizaciones

def evaluar_modelo(modelo, ds_test, clases, nombre_modelo, ruta_salida: Path):
    print(f"\n{'═'*60}")
    print(f"  EVALUACIÓN FINAL — {nombre_modelo.upper()}")
    print(f"{'═'*60}")

    y_true, y_pred_probs = [], []
    for x, y in ds_test:
        probs = modelo.predict(x, verbose=0)
        y_pred_probs.append(probs)
        y_true.append(y.numpy())

    y_true_cat  = np.concatenate(y_true, axis=0)
    y_pred_prob = np.concatenate(y_pred_probs, axis=0)
    y_true_idx  = np.argmax(y_true_cat, axis=1)
    y_pred_idx  = np.argmax(y_pred_prob, axis=1)

    accuracy    = np.mean(y_true_idx == y_pred_idx)
    top3_acc    = top_k_accuracy_score(y_true_idx, y_pred_prob, k=3)
    f1_macro    = f1_score(y_true_idx, y_pred_idx, average="macro")
    f1_weighted = f1_score(y_true_idx, y_pred_idx, average="weighted")

    print(f"\n  Accuracy          : {accuracy*100:.2f}%")
    print(f"  Top-3 Accuracy    : {top3_acc*100:.2f}%")
    print(f"  F1 Macro          : {f1_macro:.4f}")
    print(f"  F1 Weighted       : {f1_weighted:.4f}")

    print(f"\n  Reporte por clase:\n")
    print(classification_report(y_true_idx, y_pred_idx, target_names=clases))

    metricas = {
        "modelo": nombre_modelo,
        "accuracy": float(accuracy),
        "top3_accuracy": float(top3_acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }
    with open(ruta_salida / "metricas.json", "w") as f:
        json.dump(metricas, f, indent=2)

    return y_true_idx, y_pred_idx, y_pred_prob

def evaluar_modelo_tflite(ruta_tflite, ds_test, clases, nombre_modelo):
    print(f"\n{'═'*60}")
    print(f"  EVALUACIÓN TFLITE — {nombre_modelo.upper()}")
    print(f"{'═'*60}")

    # Carga intérprete TFLite
    interpreter = tf.lite.Interpreter(model_path=str(ruta_tflite))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    y_true = []
    y_pred_probs = []

    for x_batch, y_batch in ds_test:
        x_batch = x_batch.numpy()

        batch_preds = []

        # TFLite normalmente procesa una imagen a la vez
        for i in range(len(x_batch)):
            input_data = np.expand_dims(
                x_batch[i].astype(np.float32),
                axis=0
            )

            interpreter.set_tensor(
                input_details[0]['index'],
                input_data
            )

            interpreter.invoke()

            output_data = interpreter.get_tensor(
                output_details[0]['index']
            )

            batch_preds.append(output_data[0])

        y_pred_probs.append(np.array(batch_preds))
        y_true.append(y_batch.numpy())

    y_true_cat  = np.concatenate(y_true, axis=0)
    y_pred_prob = np.concatenate(y_pred_probs, axis=0)

    y_true_idx = np.argmax(y_true_cat, axis=1)
    y_pred_idx = np.argmax(y_pred_prob, axis=1)

    accuracy = np.mean(y_true_idx == y_pred_idx)

    f1_macro = f1_score(
        y_true_idx,
        y_pred_idx,
        average="macro"
    )

    print(f"\n  Accuracy TFLite : {accuracy*100:.2f}%")
    print(f"  F1 Macro        : {f1_macro:.4f}")

    print("\n  Reporte por clase:\n")
    print(classification_report(
        y_true_idx,
        y_pred_idx,
        target_names=clases
    ))

    return accuracy, f1_macro


def graficar_historial(historial_f1, historial_f2, nombre_modelo, ruta_salida: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{nombre_modelo} — Historial de entrenamiento", fontsize=14)

    epocas_f1 = range(1, len(historial_f1.history["loss"]) + 1)
    offset     = len(epocas_f1)
    epocas_f2  = range(offset + 1, offset + len(historial_f2.history["loss"]) + 1)

    for ax, metrica, titulo in zip(
        axes,
        [("accuracy", "val_accuracy"), ("loss", "val_loss")],
        ["Accuracy", "Loss"]
    ):
        train_key, val_key = metrica
        ax.plot(epocas_f1, historial_f1.history[train_key],
                "b-", label="Train Fase 1")
        ax.plot(epocas_f1, historial_f1.history[val_key],
                "b--", label="Val Fase 1")
        ax.plot(epocas_f2, historial_f2.history[train_key],
                "r-", label="Train Fase 2 (fine-tune)")
        ax.plot(epocas_f2, historial_f2.history[val_key],
                "r--", label="Val Fase 2 (fine-tune)")
        ax.axvline(x=offset + 0.5, color="gray", linestyle=":", linewidth=1.5,
                   label="Inicio fine-tuning")
        ax.set_title(titulo)
        ax.set_xlabel("Época")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta_salida / "historial_entrenamiento.png", dpi=150)
    plt.close()
    print(f" Gráfica de historial guardada.")


def graficar_confusion(y_true, y_pred, clases, nombre_modelo, ruta_salida: Path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"{nombre_modelo} — Matriz de Confusión", fontsize=14)

    for ax, normalizar, titulo in zip(
        axes,
        [False, True],
        ["Valores absolutos", "Normalizada (%)"]
    ):
        cm = confusion_matrix(y_true, y_pred,
                              normalize="true" if normalizar else None)
        fmt    = ".2f" if normalizar else "d"
        vmax   = 1.0 if normalizar else None
        sns.heatmap(
            cm, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=clases, yticklabels=clases,
            ax=ax, vmin=0, vmax=vmax,
            linewidths=0.5, linecolor="lightgray"
        )
        ax.set_title(titulo)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(ruta_salida / "matriz_confusion.png", dpi=150)
    plt.close()
    print(f"  Matriz de confusión guardada.")


def graficar_f1_por_clase(y_true, y_pred, clases, nombre_modelo, ruta_salida: Path):
    f1s = f1_score(y_true, y_pred, average=None)

    fig, ax = plt.subplots(figsize=(9, 5))
    colores = ["#e74c3c" if v < 0.7 else "#f39c12" if v < 0.85 else "#2ecc71"
               for v in f1s]
    bars = ax.bar(clases, f1s, color=colores, edgecolor="white", linewidth=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{nombre_modelo} — F1-Score por clase", fontsize=13)
    ax.set_ylabel("F1-Score")
    ax.axhline(y=np.mean(f1s), color="steelblue", linestyle="--",
               linewidth=1.5, label=f"Promedio: {np.mean(f1s):.3f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{val:.3f}", ha="center", va="bottom", fontsize=10)

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(ruta_salida / "f1_por_clase.png", dpi=150)
    plt.close()
    print(f"  F1 por clase guardado.")


def graficar_confianza(y_true, y_pred_prob, y_pred, clases,
                       nombre_modelo, ruta_salida: Path):
    confianza = np.max(y_pred_prob, axis=1)
    correctas = y_true == y_pred

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(confianza[correctas],  bins=30, alpha=0.7,
            color="#2ecc71", label="Predicción correcta")
    ax.hist(confianza[~correctas], bins=30, alpha=0.7,
            color="#e74c3c", label="Predicción incorrecta")
    ax.set_title(f"{nombre_modelo} — Confianza del modelo", fontsize=13)
    ax.set_xlabel("Probabilidad máxima (confianza)")
    ax.set_ylabel("Nº de imágenes")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta_salida / "distribucion_confianza.png", dpi=150)
    plt.close()
    print(f" Distribución de confianza guardada.")

# Entrenamiento completo

def formatear_tiempo(segundos: float) -> str:
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


def entrenar_modelo(nombre: str, ruta_data: Path, ruta_resultados: Path, seed: int):
    print(f"\n{'═'*60}")
    print(f"  INICIANDO: {nombre.upper()}")
    print(f"{'═'*60}\n")
    tiempo_inicio = time.time()

    ruta_modelo = ruta_resultados / nombre
    ruta_modelo.mkdir(parents=True, exist_ok=True)

    modelo, base, preprocesar_fn = construir_modelo(nombre, NUM_CLASES)

    factory = construir_datasets(ruta_data, seed=seed)
    ds_train, clases = factory("train", preprocesar_fn, entrenar=True)
    ds_val,   _      = factory("val",   preprocesar_fn, entrenar=False)
    ds_test,  _      = factory("test",  preprocesar_fn, entrenar=False)

    print(f"  Clases detectadas: {clases}")
    pesos_clase = calcular_pesos_clase(clases)
    print(f"  Pesos por clase  : { {clases[i]: round(v,3) for i,v in pesos_clase.items()} }\n")

    print(f" FASE 1: Entrenando cabeza ──")
    modelo.compile(
        optimizer=keras.optimizers.Adam(FASE1_LR),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=["accuracy", keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc")]
    )
    modelo.summary(print_fn=lambda x: None)

    callbacks_f1 = [
        ValScore(peso_acc=0.6, peso_loss=0.4),
        EarlyStopping(monitor="val_score", patience=7, mode="max",   # ↑ de 5: más tiempo para converger con MixUp
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, verbose=1, min_lr=1e-6),       # ↑ patience de 3→4: MixUp hace la loss más ruidosa
        ModelCheckpoint(str(ruta_modelo / "mejor_fase1.keras"),
                        monitor="val_score", save_best_only=True,
                        mode="max", verbose=0),
    ]

    hist_f1 = modelo.fit(
        ds_train,
        validation_data=ds_val,
        epochs=FASE1_EPOCHS,
        class_weight=pesos_clase,
        callbacks=callbacks_f1,
        verbose=1,
    )

    tiempo_fase1 = time.time() - tiempo_inicio
    print(f"\n  Fase 1 completada en: {formatear_tiempo(tiempo_fase1)}")
    tiempo_fase2_inicio = time.time()

    print(f"\n  ── Ajustando dropout para fase 2 ──")
    subir_dropout(modelo, DROPOUT_FASE2)

    n_capas = CAPAS_DESCONGELAR[nombre]
    print(f"\n  ── FASE 2: Fine-tuning (últimas {n_capas} capas de {nombre}) ──")
    modelo = descongelar_para_finetune(modelo, base, n_capas, FASE2_LR)

    callbacks_f2 = [
        ValScore(peso_acc=0.6, peso_loss=0.4),
        # CosineAnnealingWarmRestarts sustituye a ReduceLROnPlateau en Fase 2.
        CosineAnnealingWarmRestarts(
            lr_max=FASE2_LR, lr_min=1e-6,
            T_0=15, T_mult=1.5, decay_factor=0.6
        ),
        EarlyStopping(monitor="val_score", patience=20, mode="max",
                      restore_best_weights=True, verbose=1),
        #   Necesitamos más margen para que no pare en el punto bajo del ciclo.
        ModelCheckpoint(str(ruta_modelo / "mejor_fase2.keras"),
                        monitor="val_score", save_best_only=True,
                        mode="max", verbose=0),
    ]

    hist_f2 = modelo.fit(
        ds_train,
        validation_data=ds_val,
        epochs=FASE2_EPOCHS,
        class_weight=pesos_clase,
        callbacks=callbacks_f2,
        verbose=1,
    )

    # Guardar modelo final en Keras
    ruta_keras = str(ruta_modelo / "modelo_final.keras")
    modelo.save(ruta_keras)
    print(f"\n Modelo guardado en: {ruta_keras}")

    # Guardar modelo en tflite
    print(f"\n  ── Convirtiendo a formato TFLite ──")
    try:
        # Se genera el convertidor a partir del modelo entrenado
        converter = tf.lite.TFLiteConverter.from_keras_model(modelo)
        
        # Reducir el tamaño del modelo drásticamente mejorando el rendimiento en móviles
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_model = converter.convert()

        ruta_tflite = ruta_modelo / "modelo_final.tflite"
        with open(ruta_tflite, "wb") as f:
            f.write(tflite_model)
        print(f" Modelo TFLite guardado en: {ruta_tflite}")
        # Evaluar modelo TFLite
        evaluar_modelo_tflite(
            ruta_tflite,
            ds_test,
            clases,
            nombre
        )
    except Exception as e:
        print(f" Error al convertir a TFLite: {e}")

    y_true, y_pred, y_prob = evaluar_modelo(
        modelo, ds_test, clases, nombre, ruta_modelo
    )
    graficar_historial(hist_f1, hist_f2, nombre, ruta_modelo)
    graficar_confusion(y_true, y_pred, clases, nombre, ruta_modelo)
    graficar_f1_por_clase(y_true, y_pred, clases, nombre, ruta_modelo)
    graficar_confianza(y_true, y_prob, y_pred, clases, nombre, ruta_modelo)

    tiempo_fase2 = time.time() - tiempo_fase2_inicio
    tiempo_total = time.time() - tiempo_inicio

    print(f"\n Resumen de tiempos — {nombre.upper()}")
    print(f"     Fase 1 (cabeza)    : {formatear_tiempo(tiempo_fase1)}")
    print(f"     Fase 2 (fine-tune) : {formatear_tiempo(tiempo_fase2)}")
    print(f"     Total              : {formatear_tiempo(tiempo_total)}")

    metricas_path = ruta_modelo / "metricas.json"
    if metricas_path.exists():
        with open(metricas_path) as f:
            m = json.load(f)
        m["tiempo_fase1_seg"] = round(tiempo_fase1, 1)
        m["tiempo_fase2_seg"] = round(tiempo_fase2, 1)
        m["tiempo_total_seg"] = round(tiempo_total, 1)
        m["tiempo_fase1_fmt"] = formatear_tiempo(tiempo_fase1)
        m["tiempo_fase2_fmt"] = formatear_tiempo(tiempo_fase2)
        m["tiempo_total_fmt"] = formatear_tiempo(tiempo_total)
        with open(metricas_path, "w") as f:
            json.dump(m, f, indent=2)

    print(f"\n Resultados guardados en: {ruta_modelo}\n")
    return ruta_modelo


# Punto de entrada

def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento de clasificador de insectos con Transfer Learning"
    )
    parser.add_argument("dataset", type=str,
                        help="Ruta al dataset dividido (con carpetas train/val/test)")
    parser.add_argument("--modelo", type=str, choices=["resnet50", "mobilenet"],
                        default=None,
                        help="Modelo a entrenar. Sin este argumento entrena ambos.")
    # Semilla controlable desde la línea de comandos
    parser.add_argument("--seed", type=int, default=SEED,
                        help=f"Semilla de aleatoriedad (default: {SEED})")
    
    args = parser.parse_args()

    # Re-sembrar con la semilla que llegó por argumento
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f" Semilla activa: {seed}")

    ruta_data = Path(args.dataset).resolve()
    if not ruta_data.exists():
        print(f" La ruta no existe: {ruta_data}")
        sys.exit(1)

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_resultados = ruta_data.parent / f"resultados_{timestamp}"
    ruta_resultados.mkdir(parents=True, exist_ok=True)

    shutil.copy2(__file__, ruta_resultados / Path(__file__).name)
    print(f" Copia del script guardada en: {ruta_resultados / Path(__file__).name}")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n GPU detectada: {[g.name for g in gpus]}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n No se detectó GPU. Entrenando en CPU.")

    modelos = (
        [args.modelo] if args.modelo
        else ["resnet50", "mobilenet"]
    )

    resultados = {}
    for nombre in modelos:
        ruta = entrenar_modelo(nombre, ruta_data, ruta_resultados, seed=seed)
        resultados[nombre] = str(ruta)

    if len(modelos) == 2:
        print(f"\n{'═'*60}")
        print(f"  COMPARATIVA FINAL")
        print(f"{'═'*60}")
        for nombre in modelos:
            json_path = Path(resultados[nombre]) / "metricas.json"
            if json_path.exists():
                with open(json_path) as f:
                    m = json.load(f)
                print(f"\n  {nombre.upper()}")
                print(f"    Accuracy     : {m['accuracy']*100:.2f}%")
                print(f"    Top-3 Acc    : {m['top3_accuracy']*100:.2f}%")
                print(f"    F1 Macro     : {m['f1_macro']:.4f}")
                print(f"    F1 Weighted  : {m['f1_weighted']:.4f}")

    if len(modelos) == 2:
        print(f"\n  {'─'*40}")
        print(f"  TIEMPOS DE ENTRENAMIENTO")
        print(f"  {'─'*40}")
        tiempo_gran_total = 0
        for nombre in modelos:
            json_path = Path(resultados[nombre]) / "metricas.json"
            if json_path.exists():
                with open(json_path) as f:
                    m = json.load(f)
                if "tiempo_total_fmt" in m:
                    print(f"  {nombre.upper():<15} {m['tiempo_total_fmt']}")
                    tiempo_gran_total += m.get("tiempo_total_seg", 0)
        if tiempo_gran_total:
            print(f"  {'TOTAL AMBOS':<15} {formatear_tiempo(tiempo_gran_total)}")

    print(f"\n Todos los resultados en: {ruta_resultados}\n")


if __name__ == "__main__":
    main()