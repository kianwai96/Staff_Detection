import cv2 as cv
#import torch
from ultralytics import YOLOv10 as YOLO
import pandas as pd
from deep_sort_realtime.deepsort_tracker import DeepSort
from slowvideo import SlowVideo as Slow


# Load the YOLOv8 model (choose 'yolov8n.pt', 'yolov8s.pt', etc. for different sizes)
model = YOLO('./runs/detect/train/weights/last.pt')  # or another version of YOLOv8 (e.g., yolov8s.pt for small)

# Load the video file
input_video_path = 'C:\\Users\\user\\OneDrive\\Desktop\\Staff_Detection\\sample.mp4'
cap = cv.VideoCapture(input_video_path)

#slow = Slow('sample.mp4')

font_scale = 1
font = cv.FONT_HERSHEY_PLAIN
frameTime = 30 

out = pd.DataFrame(columns=('frame number', 'coordinates'))

deepsort = DeepSort(max_age=20, n_init=3)

#cap = cv.VideoCapture('SlowedVideo.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame,conf=0.5, iou=0.4)[0]
    
    
    detections = []
    tracks = []

    for result in results.boxes.data.tolist():
        #x1, y1, x2, y2, conf, cls = result[:6]
        xmin, ymin, xmax, ymax = int(result[0]), int(result[1]), int(result[2]), int(result[3])
        class_id = int(result[5])
        conf = result[4]
        coor = [(xmin + xmax) / 2, (ymin + ymax) / 2]
        label = f'{model.names[class_id]} {conf:.2f} ({coor[0]:.2f}, {coor[1]:.2f})'

        # Draw bounding box and label on the frame
        #if conf > 0.5:
        if class_id == 0:
            detections.append([[xmin, ymin, xmax - xmin, ymax - ymin], conf, class_id])

        frame_no = int(cap.get(cv.CAP_PROP_POS_FRAMES))
        out = pd.concat([out, pd.DataFrame({'frame number': [frame_no], 'coordinates': [coor]})], ignore_index=True)

    # Update DeepSORT tracker with the new detections

    if detections:  # Ensure detections are present
        tracks = deepsort.update_tracks(detections, frame=frame)

    for track in tracks:
        # if the track is not confirmed, ignore it
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()

        xmin, ymin, xmax, ymax = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 4)
        cv.putText(frame, f'ID: {track_id} {conf:.2f} ({coor[0]:.2f}, {coor[1]:.2f})', (xmin + 10, ymin + 10), font, font_scale, color=(0, 255, 0))

    cv.imshow('Staff Detection', frame)

    if cv.waitKey(frameTime) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

out.to_csv('results.csv')