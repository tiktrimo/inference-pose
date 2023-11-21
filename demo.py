import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np


ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])


def main():
    frame_width, frame_height = [1280, 720]

    cap = cv2.VideoCapture("./demo_infer.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    
    isTaken = False
    isOut = False
    
    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True, classes=0)[0]
        detections = sv.Detections.from_ultralytics(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f} {xyxy[0]:0.1f},{xyxy[1]:0.1f}/{xyxy[2]:0.1f},{xyxy[3]:0.1f}"
            for confidence, class_id, xyxy in zip(detections.confidence, detections.class_id, detections.xyxy)
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        for confidence, class_id, xyxy in zip(detections.confidence, detections.class_id, detections.xyxy):
            centerX = (xyxy[0] + xyxy[2]) / 2 
            centerY = (xyxy[1] + xyxy[3]) / 2
            if centerX > 500:
                isTaken = True
            else:
                isTaken = False
            print(isTaken)
        
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()