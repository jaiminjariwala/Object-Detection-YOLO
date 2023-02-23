from ultralytics import YOLO
import cv2
import cvzone
import math

model = YOLO("../YOLO-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "pen"
              ]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)    # width
cap.set(4, 720)     # height, (1280x720) or (640x480 is also common)

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # For Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 3)  ---> for openCV

            w, h = x2-x1, y2-y1
            # bbox = int(x1), int(y1), int(w), int(h)

            print(x1, y1, w, h)       # will get the values

            # for providing a fancy rectangle...
            cvzone.cornerRect(img, (x1, y1, w, h), colorC=(0, 255, 0), colorR=(0,0,0), rt=3, t=5)    # ---> for cvzone
            # colorR = color of rectangle, colorC = color of Corner, rt = rectangle thickness, t = corner thickness

            # For finding out the Confidence...
            conf = math.ceil((box.conf[0]*100))/100   # for rounding off with 2 decimal places
            print(conf)
            # cvzone.putTextRect(img, f"{conf}", (max(0, x1+5), max(35, y1-15)), thickness=2, colorR=(0,0,0), scale=2)

            # For Naming the Class...
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f"{classNames[cls]} {conf}", (max(0, x1 + 5), max(35, y1 - 15)), thickness=2, colorR=(0, 0, 0),
                               scale=2)

    cv2.imshow("Image", img)    # show image
    cv2.waitKey(1)  # 1 milli second delay


