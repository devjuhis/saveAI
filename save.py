import cv2
from ultralytics import YOLO
import os
import time

def process_video(input_video_path, output_video_path):
    # Lataa malli (YOLOv8)
    model = YOLO('best-3.pt')

    # Luo output-kansio, jos ei ole olemassa
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # Avaa video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Virhe: Videotiedostoa ei voitu avata.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Määrittele videon tallennus
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Parametrit
    default_frame_spand = 10  # Frame interval for scanning
    confidence_treshold = 0.7  # Confidence threshold for object detection
    time_before = 2  # Time (in seconds) before the action (rewinding)
    time_after = 2  # Time (in seconds) after the action
    current_frame = 0  # Current frame number
    frames_after_counter = 0  # Frames to track after detecting action
    last_written_frame = -1  # Muuttuja, joka seuraa viimeksi kirjoitetun kehyksen

    # Kohdeluokat (RVH, save, set)
    target_classes = [3, 4, 5] 

    # Aloita ajan mittaus
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or error at frame {current_frame}. Exiting.")
            break

        # Haravointi (scan) joka n. 'default_frame_spand' kehysvälin välein
        if current_frame % default_frame_spand == 0:
            print(f"Current frame: {current_frame} - Scanning for activity...")
            results = model(frame)
            confidences = results[0].boxes.conf.cpu().numpy()
            class_indices = results[0].boxes.cls.cpu().numpy().astype(int)

            action_detected = False
            for index, class_index in enumerate(class_indices):
                if confidences[index] > confidence_treshold and class_index in target_classes:
                    action_detected = True
                    print(f"Action detected at frame {current_frame} with confidence {confidences[index]:.2f}.")
                    break

            # Jos aktiivisuus havaitaan
            if action_detected:
                target_frame = max(current_frame - int(fps * time_before), 0)
                print(f"Rewinding to frame {target_frame} (approx. {time_before} seconds back).")
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)  # Siirry haluttu aika taaksepäin
                current_frame = target_frame  # Aseta nykyinen kehys menneisyyteen

                frames_before_counter = int(fps * time_before)  # Kehysten määrä ennen aktiivisuutta
                frames_after_counter = int(fps * time_after)  # Kehysten määrä aktiivisuuden jälkeen

                # Tiukka tarkastussilmukka (scanning after event)
                while frames_before_counter > 0 or frames_after_counter > 0:
                    ret, frame = cap.read()
                    if not ret:
                        print(f"End of video or error during detailed analysis at frame {current_frame}.")
                        break

                    # Tarkista, onko kehys jo kirjoitettu
                    if current_frame <= last_written_frame:
                        print(f"Frame {current_frame} already written. Skipping...")
                        frames_before_counter -= 1
                        frames_after_counter -= 1
                        current_frame += 1
                        continue

                    # Kirjoita kehys aina ulostulovideoon
                    text = f"Frame: {current_frame}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]  # Laskee tekstin koon
                    text_x = (frame_width - text_size[0]) // 2  # Keskittää tekstin vaakasuunnassa
                    text_y = frame_height - 30  # Siirtää tekstin lähemmäs alareunaa

                    cv2.putText(frame, text, (text_x, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    out.write(frame)
                    last_written_frame = current_frame  # Päivitä viimeksi kirjoitettu kehys
                    print(f"Writing frame {current_frame} to output video.")

                    # Päivitä nykyinen kehys
                    current_frame += 1
                    print(f"Processing frame {current_frame}.")

                    # Päivitä laskuri
                    frames_before_counter -= 1
                    print(f"Frames before counter: {frames_before_counter}, Frames after counter: {frames_after_counter}")

                    # Tarkista, onko uusi aktiviteetti havaittu (resetoi after_counter)
                    results = model(frame)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_indices = results[0].boxes.cls.cpu().numpy().astype(int)

                    action_detected = False
                    for index, class_index in enumerate(class_indices):
                        if confidences[index] > confidence_treshold and class_index in target_classes:
                            action_detected = True
                            frames_after_counter = int(fps * time_after)  # Nollaa after-counter
                            print(f"Additional action detected at frame {current_frame}. Resetting after-counter.")
                            break

                    # Jos aika ennen alkuperäistä torjuntaa on loppunut, vähennetään after-counteria
                    if frames_before_counter <= 0:
                        frames_after_counter -= 1
                        print(f"Decrementing after-counter: {frames_after_counter}")

                    # Lopeta tarkastelu, kun aktiivisuus päättyy ja frames_after_counter täyttyy
                    if frames_after_counter <= 0 and not action_detected:
                        print(f"No further activity. Ending detection loop at frame {current_frame}.")
                        break

                # Palaa skannaustilaan
                print(f"Returning to scanning mode at frame {current_frame}.")
                continue

        # Siirry seuraavaan kehykseen haravoinnissa
        current_frame += 1
        print(f"Advancing to frame {current_frame}.")

    # Lopeta ajan mittaus
    elapsed_time = time.time() - start_time
    print(f"Ohjelma kesti {int(elapsed_time // 60)} minuuttia ja {int(elapsed_time % 60)} sekuntia.")
    print(f"Tallennetaan video tiedostoon: {output_video_path}")
    print(f"Resoluutio: {frame_width}x{frame_height}, FPS: {fps}")

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Esimerkki funktion käytöstä
input_video = '26.12.2024_20.26.55_REC.mp4'
output_video = './output/clipped_saveAI_video.mp4'
process_video(input_video, output_video)
