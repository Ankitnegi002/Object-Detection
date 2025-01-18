# Libraries used in the code
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to get image path
def get_image_path():
    Tk().withdraw()  
    file_path = askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")])
    return file_path

# Load YOLO model
yolo = cv2.dnn.readNet(r"D:\Object_Detection\yolov3.weights", r"D:\Object_Detection\yolov3.cfg")


# Load COCO classes
obj = []
with open(r"D:\Object_Detection\coco.names", 'r') as f:
    obj = f.read().splitlines()

# Get image path from user
image_path = get_image_path()
img = cv2.imread(image_path)
height, width, channels = img.shape

# Preprocess image for YOLO
preprocessed_image = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)

# Perform forward pass
yolo.setInput(preprocessed_image)
detection_layers = yolo.getUnconnectedOutLayersNames()
detections = yolo.forward(detection_layers)

# Post-processing
boxes = []
probab = []
obj_ids = []
class_counts = {}  

for output in detections:
    for detect in output:
        scores = detect[5:]
        obj_id = np.argmax(scores)
        confidence = scores[obj_id]

        if confidence > 0.7:  
            center_x = int(detect[0] * width)
            center_y = int(detect[1] * height)
            w = int(detect[2] * width)
            h = int(detect[3] * height)
            # Center value of x and y
            x = int(center_x - (w / 2))
            y = int(center_y - (h / 2))

            boxes.append([x, y, w, h])
            probab.append(float(confidence))
            obj_ids.append(obj_id)

            # Count the detected objects
            class_name = obj[obj_id]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

# Calculate statistics
total_detections = len(boxes)
average_confidence = np.mean(probab) if probab else 0

# Display statistics in the console
print("\n--- Detection Statistics ---")
print(f"Total detections: {total_detections}")
print(f"Average confidence: {average_confidence:.3f}")
print("\nDetected objects:")
for class_name, count in class_counts.items():
    print(f"- {class_name}: {count}")

# Apply Non-Maximum Suppression (NMS)
indexes = cv2.dnn.NMSBoxes(boxes, probab, 0.5, 0.4)

# Draw bounding boxes
font = cv2.FONT_HERSHEY_PLAIN 
colors = np.random.randint(0, 255, size=(len(obj), 3), dtype='uint8')  

for i in indexes.flatten():
    x, y, w, h = boxes[i]

    label = str(obj[obj_ids[i]])
    confi = str(round(probab[i], 2))
    color = [int(c) for c in colors[obj_ids[i]]]

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, f"{label} {confi}", (x, y - 10), font, 2, (255, 255, 255), 2)

# Display the output image
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
