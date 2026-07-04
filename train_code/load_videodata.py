import cv2
from ultralytics import YOLO


#1.视频窗口
video_path= "material/skycraper4.mp4"
window_name="runs/detect/yolov8_fire_attn_newdata-3/weights/best.pt"
conf_thresh=0.2
cap = cv2.VideoCapture(video_path)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
paused = False

#2.加载模型
model=YOLO("yolo26s.pt")

#3.创建可缩放窗口
cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name,w,h)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #yolo推理
    results = model(frame, conf=conf_thresh)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {conf:.2f}"


            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if paused:
        cv2.putText(frame, "PAUSED (SPACE to resume)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(window_name,frame)
    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

cap.release()
cv2.destroyAllWindows()