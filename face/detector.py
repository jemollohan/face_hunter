import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

from frame_counter import FrameCounter
from factory.logger_factory import get_logger

logger = get_logger("detector")

# COCO class labels (MobileNet SSD is typically trained on COCO)
# The model output will give class IDs. '1' usually corresponds to 'person'.
# You can find the full COCO label map online. For this project, we only need 'person'.
PERSON_CLASS_ID = 1
CONFIDENCE_THRESHOLD = 0.5 # Only detect persons with confidence > 50%

def convert_to_tensor(frame):
    tensor = tf.convert_to_tensor(frame)
    return tensor[tf.newaxis, ...]

def detect(frame, detection_model):

    input_tensor = convert_to_tensor(frame)

    # Perform detection
    detections = detection_model(input_tensor)
    return detections

def extract_detections(detections):
    # Extract results
    # The output structure can vary slightly between models.
    # Refer to the model's documentation on TF Hub.
    # Typically, you get boxes, classes, and scores.
    num_detections = int(detections.pop('num_detections'))

    # Squeeze the batch dimension and select up to num_detections
    processed_detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    processed_detections['num_detections'] = num_detections
    processed_detections['detection_classes'] = processed_detections['detection_classes'].astype(np.int64)
    
    return processed_detections

def get_bbox_score_list(detections, frame_width, frame_height):

    person_boxes = []
    person_scores = []

    for i in range(len(detections['detection_scores'])):
        score = detections['detection_scores'][i]
        class_id = detections['detection_classes'][i]

        if class_id == PERSON_CLASS_ID and score > CONFIDENCE_THRESHOLD:
            # Bounding box coordinates are usually normalized (0.0 to 1.0)
            # Convert them to pixel coordinates
            ymin, xmin, ymax, xmax = detections['detection_boxes'][i]
            (left, right, top, bottom) = (xmin * frame_width, xmax * frame_width,
                                          ymin * frame_height, ymax * frame_height)
            person_boxes.append([int(left), int(top), int(right-left), int(bottom-top)]) # x, y, w, h
            person_scores.append(float(score))

    return person_boxes, person_scores

def draw_detections(frame, boxes, scores, fps=0):
    for i, box in enumerate(boxes):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Person: {scores[i]:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "Count: {}".format(fps), (10,10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    return frame



def record_video(frame, video_writer=None):
    if frame is not None:
        video_writer.write(frame)


# --- Main Loop for Detection Only (Phase 1) ---
if __name__ == "__main__": # Placeholder for now, will integrate tracking later
    pass

