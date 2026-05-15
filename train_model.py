# Päivitetään pip ja asennetaan tarvittavat kirjastot
#!pip install --upgrade pip
#!pip install ultralytics roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="your api key here")
project = rf.workspace("saveaiproto").project("saveai")
version = project.version(4)
dataset = version.download("yolov8")

from ultralytics import YOLO

# Lataa YOLOv8-malli
model = YOLO("bestv4.pt")  # Esikoulutettu malli"

# Koulutetaan malli lisäparametreilla
model.train(
    data="/content/saveAI-4/data.yaml",  # Datasetin polku
    epochs=100,                          # Epoch-määrä
    imgsz=640,                           # Kuvien koko pikseleinä
    batch=16,                            # Eräkoko
    device=0,                            # GPU:n käyttö, "0" = käytä ensimmäistä GPU:ta
)
