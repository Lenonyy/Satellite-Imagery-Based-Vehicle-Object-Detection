
from ultralytics import YOLO
import os

model = YOLO('yolov8s.pt')

input_folder = "uydu_gorselleri"
output_folder = "sonuc"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        results = model(img_path)
        output_path = os.path.join(output_folder, f"tespit_{filename}")
        results[0].save(filename=output_path)
        print(f"Tespit tamamlandı: {output_path}")
