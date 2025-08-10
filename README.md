# Multicámara con detección facial en Python

Proyecto para recibir vídeo de múltiples clientes con webcam, procesar detección y reconocimiento facial usando OpenCV LBPH.

## Requisitos

- Python 3.x
- OpenCV con contrib: `pip install opencv-contrib-python numpy`

## Estructura

- `entrenar_lbph.py`: Script para entrenar modelo con dataset.
- `servidor_multicamara.py`: Servidor que recibe vídeo y detecta caras.
- `cliente_simple.py`: Cliente que envía vídeo capturado de la webcam.
- `dataset/`: Carpeta para imágenes de entrenamiento (no subida).

## Uso

1. Preparar `dataset` con carpetas por persona y fotos dentro.
2. Ejecutar entrenamiento:

```bash
python entrenar_lbph.py
