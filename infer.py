from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator

model=YOLO("yolov8m.pt")

result = model.predict(source="./demos/C_3_12_39_BU_SMC_10-14_11-41-49_CA_RGB_DF2_M2.mp4", 
                       show=True, conf=0.001, iou=0, line_width=1)


for r in result:
    coord = result.boxes.xyxy
    print(coord)
