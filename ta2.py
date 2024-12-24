from ultralytics import YOLO
import cv2

# Load a YOLO model
model = YOLO("yolo11n.pt")

# Specify the classes you want to detect
desired_classes = [0, 2, 16]  # Example: 0 for person, 2 for car, etc.

# Path to the video file
video_path = "home/ desktop/project "  

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer to save the results
out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model.predict(frame, classes=desired_classes, conf=0.5, verbose=False, device=0)

    # Process each result
    for result in results:
        annotated_frame = result.plot()

    # Display the annotated frame
    cv2.imshow("YOLO Detection", annotated_frame) # NOTE: If lagging, comment this line

    # Write the frame to the output video
    out.write(annotated_frame)

    # Press 'q' to exit the video display 
    if cv2.waitKey(1) & 0xFF == ord('q'): # NOTE: If lagging, comment this if statement
        break

# Release video objects and close display window
cap.release()
out.release()
cv2.destroyAllWindows()
