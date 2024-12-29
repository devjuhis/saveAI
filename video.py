import cv2
from ultralytics import YOLO
import os
import time

def process_video(input_video_path, output_video_path):
    # Lataa YOLO-malli
    model = YOLO('bestv4.pt')

    # Luo output-kansio, jos ei ole olemassa
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Avaa video
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Kehyksen leveys
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Kehyksen korkeus
    fps = cap.get(cv2.CAP_PROP_FPS)  # Videon FPS

    # Määrittele videon tallennus
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Käytetään mp4-formaattia
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Halutut luokat (indexit)
    target_classes = [0, 1, 2, 3, 4, 5]  # kaikki luokat v4 mallissa

    # Aloita ajan mittaus
    start_time = time.time()

    # Käy videon kehykset läpi
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Suorita objektien tunnistus jokaisessa kehyksessä
        results = model(frame)

        # Hae tunnistetut objektit
        boxes = results[0].boxes.xywh.cpu().numpy()  # Bounding box -koordinaatit
        confidences = results[0].boxes.conf.cpu().numpy()  # Luottamustasot
        class_indices = results[0].boxes.cls.cpu().numpy().astype(int)  # Luokkaindeksit
        labels = results[0].names  # Luokkien nimet

        # Käy läpi tunnistetut objektit
        for i, box in enumerate(boxes):
            if confidences[i] > 0.7 and class_indices[i] in target_classes:
                # Muunna koordinaatit piirrettävään muotoon
                x1, y1, w, h = box
                x1, y1, x2, y2 = int(x1 - w / 2), int(y1 - h / 2), int(x1 + w / 2), int(y1 + h / 2)

                # Piirrä neliö ympärille
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Kirjoita luokan nimi, indeksi ja luottamusprosentti neliön viereen
                label = f'{labels[class_indices[i]]} ({class_indices[i]}) {confidences[i]:.2f}'
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Kirjoita kehys output-videoon
        out.write(frame)

    # Lopeta ajan mittaus
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Tulosta kulunut aika minuutteina ja sekunteina
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Ohjelma kesti {minutes} minuuttia ja {seconds} sekuntia.")

    # Vapauta resurssit
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Esimerkki käytöstä
input_video = 'kespoou18.mp4'
output_video = './output/boxed_video.mp4'
process_video(input_video, output_video)
