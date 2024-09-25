import cv2 as cv

class SlowVideo:
    def __init__(self, path):
        # Load the video file
        cap = cv.VideoCapture(path)

        # Get video properties
        height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv.CAP_PROP_FPS)

        # Set up output video writer
        output_path = "SlowedVideo.mp4"
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        
        # Set the slowed FPS (e.g., 2x slower: fps divided by 2)
        new_fps = fps // 2
        output = cv.VideoWriter(output_path, fourcc, new_fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Write each frame twice to slow down the video (2x slower)
            output.write(frame)  # Write the frame once
            output.write(frame)  # Write the same frame again to slow down

        cap.release()
        output.release()
        print(f"Video has been slowed down and saved at {output_path}")

