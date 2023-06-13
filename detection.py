import cv2
import torch
import matplotlib.pyplot as plt
import datetime

current_time = datetime.datetime.now()

# Load the YOLOv5 model
model = torch.hub.load("ultralytics/yolov5", "custom", path="best.pt")

# Set the confidence threshold
confidence_threshold = 0.5

# Open the video file
video_path = "./yolov5/goldfish.mp4"
video = cv2.VideoCapture(video_path)

# Define output video settings
output_path = "output.mp4"
output_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_video = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))

textLabel = ""

while True:
    # Read the current frame
    ret, frame = video.read()
    if not ret:
        break

    # Convert frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(frame_rgb)

    # Extract bounding boxes, labels, and scores from results
    boxes = results.xyxy[0][:, :4].tolist()
    labels = results.xyxy[0][:, 5].tolist()
    scores = results.xyxy[0][:, 4].tolist()

    # Filter detections based on confidence threshold
    filtered_boxes = []
    filtered_labels = []
    filtered_scores = []

    for box, label, score in zip(boxes, labels, scores):
        if score > confidence_threshold:
            filtered_boxes.append(box)
            filtered_labels.append(label)
            filtered_scores.append(score)

    # Render filtered detections on the frame
    rendered_frame = frame.copy()

    for box, label, score in zip(filtered_boxes, filtered_labels, filtered_scores):
        current_time = datetime.datetime.now()
        if label == 0.0:
            textLabel = "Goldfish"
        x1, y1, x2, y2 = box
        cv2.rectangle(
            rendered_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
        )

        # Calculate the centroid coordinates
        centroid_x = (x1 + x2) // 2
        centroid_y = (y1 + y2) // 2

        # Draw background rectangle for text
        text = f"{textLabel}: {score:.2f}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(
            rendered_frame,
            (int(x1), int(y1)),
            (int(x1) + text_size[0] + 10, int(y1) - text_size[1] - 10),
            (0, 255, 0),
            cv2.FILLED,
        )

        # Overlay text on top of the background rectangle
        cv2.putText(
            rendered_frame,
            text,
            (int(x1) + 5, int(y1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2,
        )

        # Print the label and centroid coordinates
        #         print(f'{label}: centroid_x={centroid_x}, centroid_y={centroid_y}')
        if centroid_y < 200:
            print(f"check your goldfish now!, time={current_time}")

    # Write the frame to the output video
    output_video.write(rendered_frame)

    # Display the frame
    cv2.imshow("Frame", rendered_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord("q"):
        break

# Release the video capture and writer
video.release()
output_video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
