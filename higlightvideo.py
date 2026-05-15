## pip install opencv-python ultralytics torch

import cv2
from ultralytics import YOLO
import os
import time
import torch

DEBUG = True # Poistaa console viestit, true laittaa takaisin

def debug_print(message):
    if DEBUG:
        print(message)

def process_video(input_video_path, output_video_path):
    # Tarkista, onko GPU käytettävissä
    device = "cuda" if torch.cuda.is_available() else "cpu"
    debug_print(f"Käytettävä laite: {device}")

    # Lataa malli ja määritä käytettävä laite
    model = YOLO('/content/drive/MyDrive/bestv4.pt')

    model.to(device)

    # Luo output-kansio, jos ei ole olemassa
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Avaa video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        debug_print("Virhe: Videotiedostoa ei voitu avata.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Määrittele videon tallennus
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Parametrit
    default_frame_spand = 10  # Kehysväli skannaukseen
    confidence_threshold = 0.7  # Luottamusraja objektin tunnistukselle
    time_before = 2  # Aika (sekunteina) ennen toimintaa (takaperoisesti)
    time_after = 1  # Aika (sekunteina) toiminnan jälkeen
    current_frame = 0  # Nykyinen kehys
    frames_after_counter = 0  # Kehykset, jotka tarkistetaan toiminnan jälkeen
    last_written_frame = -1  # Viimeksi kirjoitettu kehys

    # Kohdeluokat (save)
    target_classes = [4]

    # Aloita ajan mittaus
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            debug_print(f"End of video or error at frame {current_frame}. Exiting.")
            break

        # Haravointi joka n. 'default_frame_spand' kehysvälin välein
        if current_frame % default_frame_spand == 0:
            debug_print(f"Current frame: {current_frame} - Scanning for activity...")
            # Lisää verbose=False mallin kutsuun
            results = model(frame, verbose=False)
            confidences = results[0].boxes.conf.cpu().numpy()
            class_indices = results[0].boxes.cls.cpu().numpy().astype(int)

            action_detected = False
            for index, class_index in enumerate(class_indices):
                if confidences[index] > confidence_threshold and class_index in target_classes:
                    action_detected = True
                    debug_print(f"Action detected at frame {current_frame} with confidence {confidences[index]:.2f}.")
                    break

            # Jos aktiivisuus havaitaan
            if action_detected:
                target_frame = max(current_frame - int(fps * time_before), 0)
                debug_print(f"Rewinding to frame {target_frame} (approx. {time_before} seconds back).")
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                current_frame = target_frame

                frames_before_counter = int(fps * time_before)
                frames_after_counter = int(fps * time_after)

                while frames_before_counter >= 0 or frames_after_counter >= 0:
                    ret, frame = cap.read()
                    if not ret:
                        debug_print(f"End of video or error during detailed analysis at frame {current_frame}.")
                        break

                    if current_frame <= last_written_frame:
                        debug_print(f"Frame {current_frame} already written. Skipping...")
                        frames_before_counter -= 1
                        frames_after_counter -= 1
                        current_frame += 1
                        continue

                    # Kirjoita kehys
                    text = f"Frame: {current_frame}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    text_x = (frame_width - text_size[0]) // 2
                    text_y = frame_height - 30
                    cv2.putText(frame, text, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    out.write(frame)
                    last_written_frame = current_frame
                    debug_print(f"Writing frame {current_frame} to output video.")

                    current_frame += 1
                    frames_before_counter -= 1

                    # Tarkista uusi aktiivisuus
                    results = model(frame, verbose=False)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_indices = results[0].boxes.cls.cpu().numpy().astype(int)

                    action_detected = False
                    for index, class_index in enumerate(class_indices):
                        if confidences[index] > confidence_threshold and class_index in target_classes:
                            action_detected = True
                            frames_after_counter = int(fps * time_after)
                            debug_print(f"Additional action detected at frame {current_frame}. Resetting after-counter.")
                            break

                    if frames_before_counter <= 0:
                        frames_after_counter -= 1

                    if frames_after_counter <= 0 and not action_detected:
                        debug_print(f"No further activity. Ending detection loop at frame {current_frame}.")
                        break

                debug_print(f"Returning to scanning mode at frame {current_frame}.")
                continue

        current_frame += 1
        debug_print(f"Advancing to frame {current_frame}.")

    elapsed_time = time.time() - start_time
    print(f"Ohjelma kesti {int(elapsed_time // 60)} minuuttia ja {int(elapsed_time % 60)} sekuntia.")
    debug_print(f"Tallennetaan video tiedostoon: {output_video_path}")
    debug_print(f"Resoluutio: {frame_width}x{frame_height}, FPS: {fps}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Esimerkki funktion käytöstä
input_video = '/content/drive/MyDrive/editoimattomat_pelit/game.mp4' # kopioi pelin polku tähän
output_video = '/content/drive/MyDrive/editoidut_pelit/editoitu_peli_nimea_uudelleen.mp4'
process_video(input_video, output_video)
