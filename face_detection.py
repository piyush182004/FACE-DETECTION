import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Load the pre-trained MobileNet SSD model
model = hub.load("https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1")

def detect_faces(image):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]  # Add batch dimension

    # Perform inference
    detections = model(input_tensor)
    return detections

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    detections = detect_faces(rgb_frame)

    # Process detections
    for i in range(len(detections['detection_boxes'])):
        score = detections['detection_scores'][i].numpy()
        if score > 0.5:  # Confidence threshold
            box = detections['detection_boxes'][i].numpy()
            h, w, _ = frame.shape
            ymin, xmin, ymax, xmax = (box * np.array([h, w, h, w])).astype(int)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
