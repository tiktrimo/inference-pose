import cv2
import json
from os import listdir
from os.path import isfile, join

from ultralytics import YOLO

def getAnnotation(filename):
    with open(filename) as fileStream:
        jsonData = json.load(fileStream)
        return jsonData
    
def getCoordination(jsonData, filename):
    imageID = -1
    for image in jsonData["images"]:
        if(filename in image["file_name"]):
            imageID = image["id"]
            break
    if(imageID == -1): return None
    
    coordination = [-1, 0, 0, 0]
    for annotation in jsonData["annotations"]:
        if(annotation["image_id"] == imageID):
            coordination[0] = annotation["bbox"][0]
            coordination[1] = annotation["bbox"][1]
            coordination[2] = annotation["bbox"][2]
            coordination[3] = annotation["bbox"][3]
            break
    if(coordination[0] == -1): return None
    
    return coordination


# draw text on the frame
def drawText(img, text, pos=(0, 0),
          font=cv2.FONT_ITALIC,
          font_scale=1,
          font_thickness=2,
          text_color=(255, 255, 255),
          text_color_bg=(96, 96, 255)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

def showPositionHPE(frame, keypoints):
    # coordinations of hands, foot
    leftHandX, leftHandY = keypoints[10]
    rightHandX, rightHandY = keypoints[9]
    footX, footY = keypoints[15]
    
    # draw text on the frame(photo)
    drawText(frame, 
                f"{int(leftHandX)}, {int(leftHandY)}", 
                (int(leftHandX),int(leftHandY)))
    drawText(frame, 
                f"{int(rightHandX)}, {int(rightHandY)}", 
                (int(rightHandX),int(rightHandY)))
    drawText(frame, 
                f"{int(footX)}, {int(footY)}", 
                (int(footX),int(footY)))
    return frame

def showOverlayDistance(frame, keypoints, register):
    # coordinations of hands, foot
    leftHandX, leftHandY = keypoints[10]
    rightHandX, rightHandY = keypoints[9]
    footX, footY = keypoints[15]
    
    frame = cv2.circle(frame, register, 5, (96, 96, 255), 5)
    if leftHandX > 1 and leftHandY > 1:
        cv2.line(frame, register, (int(leftHandX), int(leftHandY)), (96, 96, 255), 5)
        drawText(frame, 
                 f"{int(getDistance(register, (int(leftHandX), int(leftHandY))))}", 
                 (int((leftHandX + register[0]) / 2), int((leftHandY + register[1]) / 2)))
        
    if rightHandX > 1 and rightHandY > 1:
        cv2.line(frame, register, (int(rightHandX), int(rightHandY)), (96, 96, 255), 5)
        drawText(frame, 
                 f"{int(getDistance(register, (int(rightHandX), int(rightHandY))))}", 
                 (int((rightHandX + register[0]) / 2), int((rightHandY + register[1]) / 2)))
        
    if footX > 1 and footY > 1:
        cv2.line(frame, register, (int(footX), int(footY)), (96, 96, 255), 5)
        drawText(frame, 
                 f"{int(getDistance(register, (int(footX), int(footY))))}", 
                 (int((footX + register[0]) / 2), int((footY + register[1]) / 2)))
    
    return

def showOverlayHPE(frame, keypoints):
    def connectPoints(frame, kp1, kp2):
        pt1 = (int(keypoints[kp1][0]), int(keypoints[kp1][1]))
        pt2 = (int(keypoints[kp2][0]), int(keypoints[kp2][1]))
        if(pt1[0] < 1 or pt2[0] < 1 or pt1[1] < 1 or pt2[1] < 1): return
        cv2.line(frame, pt1, pt2, (255, 255, 255), 2)
        
    connectPoints(frame, 0, 2)
    connectPoints(frame, 0, 1)
    connectPoints(frame, 2, 4)
    connectPoints(frame, 1, 3)
    connectPoints(frame, 3, 5)
    connectPoints(frame, 4, 6)
    
    connectPoints(frame, 5, 6)
    connectPoints(frame, 6, 12)
    connectPoints(frame, 6, 8)
    connectPoints(frame, 8, 10)
    connectPoints(frame, 5, 7)
    connectPoints(frame, 5, 11)
    connectPoints(frame, 11, 12)
    connectPoints(frame, 7, 9)
    connectPoints(frame, 12, 14)  
    connectPoints(frame, 14, 16)
    connectPoints(frame, 11, 13)
    connectPoints(frame, 13, 15)
    

    return

def getDistance(register_pos, point):
    x_diff = pow(register_pos[0] - point[0], 2)
    y_diff = pow(register_pos[1] - point[1], 2)
    distant = pow(x_diff + y_diff, 0.5)
    
    return distant

def getRegister(coordination, resize=1):
    return (int((coordination[0] + coordination[2] / 2) * resize), int((coordination[1] + coordination[3] / 2)* resize))


def main():
    VIDEO_DIRECTORY = "./demos/"
    
    pose_model = YOLO("yolov8n-pose.pt")
    object_model = YOLO("yolov8n.pt")
    videos= [f for f in listdir(VIDEO_DIRECTORY) if isfile(join(VIDEO_DIRECTORY, f))]
    
    DISTANT_THRESHOLD = 200
    TIME_THRESHOLD = 60
    EXAMPLE_JSON_FILE = 'annotation_A.json'
    
    annotation = getAnnotation(EXAMPLE_JSON_FILE)
    
    validVideoCount = 0
    theifVideoCount = 0
    for progress, video in enumerate(videos):      
        coordination = getCoordination(annotation, video)
        if(coordination == None): 
            print(f"{progress}/{len(videos)} |+| Invalid")
            continue
        
        cap = cv2.VideoCapture(f"{VIDEO_DIRECTORY}{video}")
        width  = cap.get(3)
        height = cap.get(4)
        
        register = getRegister(coordination, width/1920)
        
        isTheif = True
        isValidVideo = True
        registerStayCount = 0
        while True:
            captureValid, frame = cap.read()
            if(not captureValid): break
            
            pose_prediction = pose_model(frame, verbose=False)
            # obj_prediction = object_model(frame, verbose=False, conf=0.001, iou=0)
            keypoints = pose_prediction[0].keypoints.xy.tolist()[0]
            if len(pose_prediction) > 1:
                isValidVideo = False
                break
            if len(keypoints) < 15: continue

            #################################################### DETECTION
            footDistance = getDistance(register, keypoints[15])
            lhandDistance = getDistance(register, keypoints[10])
            rhandDistance = getDistance(register, keypoints[9])
            if(footDistance < DISTANT_THRESHOLD + 200 or lhandDistance < DISTANT_THRESHOLD or rhandDistance < DISTANT_THRESHOLD): registerStayCount += 1
            else : registerStayCount = 0
            
            if registerStayCount > TIME_THRESHOLD:
                isTheif = False
                break
            
            #################################################### OVERLAY
            # frame = results[0].plot()
            showOverlayHPE(frame, keypoints)
            showOverlayDistance(frame, keypoints, register)
            cv2.imshow("nogosan_window_name", frame)

            # press esc to quit
            if (cv2.waitKey(30) == 27): break
        if isValidVideo: validVideoCount += 1
        if isTheif and isValidVideo: theifVideoCount += 1
        print(f"{progress}/{len(videos)} |+| {theifVideoCount}/{validVideoCount}")
        
    print(f"{theifVideoCount}/{validVideoCount}")


if __name__ == "__main__":
    main()