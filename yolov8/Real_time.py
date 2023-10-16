import cv2
import os
from ultralytics import YOLO


cap = cv2.VideoCapture("http://192.168.8.1:8083/?action=snapshot")

# Create directory to save annotated frames
save_dir = "./output"
os.makedirs(save_dir, exist_ok=True)

model = YOLO("best27.pt")

frame_counter = 0  # Frame counter
save_interval = 1  # 保存图像的间隔帧数
frame_skip = 0  # 初始化帧跳过计数器

# Loop through the video frames
# while cap.isOpened():
while True:
    # if neen_to_reconnect():
    #     reconnect_vedio_stream()
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame_skip += 1

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        if frame_skip >= save_interval:
            # Save the annotated frame as an image
            save_path = os.path.join(save_dir, f"frame_{frame_counter}.jpg")
            cv2.imwrite(save_path, annotated_frame)

            frame_counter += 1
            frame_skip = 0

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()