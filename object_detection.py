import cv2 as cv
#import torch
from ultralytics import YOLOv10 as YOLO
import pandas as pd

# Load the YOLOv8 model (choose 'yolov8n.pt', 'yolov8s.pt', etc. for different sizes)
model = YOLO('./runs/detect/train/weights/last.pt')  # or another version of YOLOv8 (e.g., yolov8s.pt for small)

# Load the video file
input_video_path = 'C:\\Users\\user\\OneDrive\\Desktop\\Staff_Detection\\sample.mp4'
cap = cv.VideoCapture(input_video_path)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out_video = cv.VideoWriter('out.mp4', cv.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))

font_scale = 1
font = cv.FONT_HERSHEY_PLAIN

out = pd.DataFrame(columns=('frame number', 'coordinates'))

while True:
    ret, frame = cap.read()

    results = model(frame)[0]

    for result in results.boxes.data.tolist():  # Each detection in the format [x1, y1, x2, y2, conf, class]
        x1, y1, x2, y2, conf, cls = result[:6]
        coor = [(x1 + x2) / 2, (y1 + y2) / 2]
        label = f'{model.names[cls]} {conf:.2f} ({coor[0]:.2f}, {coor[1]:.2f})'

            # Draw bounding box and label on the frame
        if conf > 0.5: 
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)  
            cv.putText(frame,label, (int(x1) + 10, int(y1) + 40), font, font_scale, color=(0,255,0))
            frame_no = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            out = out.append({'frame number': frame_no, 'coordinates': coor}, ignore_index=True)
    
    cv.imshow('Staff Detection', frame)
    out_video.write(frame)
    if cv.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
out_video.release()
cv.destroyAllWindows()

out.to_csv('results.csv')

