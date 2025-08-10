import cv2
import os
import numpy as np

def get_images_and_labels(dataset_path):
    faces = []
    labels = []
    label_map = {}  # nombre_persona -> id

    current_label = 0

    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_folder):
            continue
        label_map[current_label] = person_name

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces.append(img)
            labels.append(current_label)

        current_label += 1

    return faces, labels, label_map

dataset_path = 'dataset'  # Cambia por tu ruta

faces, labels, label_map = get_images_and_labels(dataset_path)

# Crear reconocedor LBPH
recognizer = cv2.face.LBPHFaceRecognizer_create()

print("[INFO] Entrenando modelo...")
recognizer.train(faces, np.array(labels))
print("[INFO] Entrenamiento finalizado.")

# Guardar modelo para usar después
recognizer.save("modelo_lbph.yml")

# Guardar mapping para etiquetas
import pickle
with open("label_map.pkl", "wb") as f:
    pickle.dump(label_map, f)
