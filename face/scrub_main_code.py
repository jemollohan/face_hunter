import cv2
import tensorflow as tf
import threading
import queue
import time
import numpy as np

# --- Configuration ---
VIDEO_SOURCE = 0  # Webcam
MODEL_PATH = "path/to/your/tensorflow/saved_model" # Or .pb, .tflite, etc.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_QUEUE_SIZE = 10 # Max frames to buffer

# --- Shared Resources ---
frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE) # For frames from camera to detector
processed_frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE) # For frames from detector to display
stop_event = threading.Event()

# --- Placeholder for your TensorFlow Model Loading ---
# In a real scenario, load your model here.
# For this example, we'll simulate it.
#
# Example:
# try:
#     detection_model = tf.saved_model.load(MODEL_PATH)
# except Exception as e:
#     print(f"Error loading TensorFlow model: {e}")
#     exit()
#
# def run_inference(model, image_np):
#     # Actual inference logic
#     # input_tensor = tf.convert_to_tensor(image_np)
#     # input_tensor = input_tensor[tf.newaxis, ...]
#     # detections = model(input_tensor)
#     # For demonstration, let's just return the image and mock detections
#     mock_detections = [{"box": [0.1, 0.1, 0.5, 0.5], "class": "dummy", "score": 0.9}]
#     return image_np, mock_detections

# --- Frame Capture Thread ---
def frame_capture_thread_func(video_source, width, height):
    print("Starting frame capture thread...")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        stop_event.set() # Signal other threads to stop
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Warning: Could not read frame, retrying or ending stream...")
            time.sleep(0.1) # Wait a bit before retrying or if stream ended
            # If you want to stop on stream end, uncomment below
            # stop_event.set()
            # break
            continue

        try:
            # Pre-processing if needed before putting in queue
            frame_queue.put(frame, timeout=1)  # Add to queue, with timeout
        except queue.Full:
            # print("Frame queue is full, dropping frame.")
            time.sleep(0.01) # Give some time for the queue to clear
            pass # Or handle differently
        except Exception as e:
            print(f"Error in frame capture: {e}")
            break

    cap.release()
    print("Frame capture thread finished.")

# --- Object Detection Thread ---
def object_detection_thread_func():
    print("Starting object detection thread...")
    # --- Load your TensorFlow model HERE ---
    # This is crucial. Model loading can be slow, so do it inside the thread
    # if you want startup to feel faster, or outside if you want to ensure
    # it's loaded before this thread even starts processing.
    # For this example, we'll use a placeholder.
    print("Simulating TensorFlow model loading...")
    time.sleep(2) # Simulate model load time
    print("TensorFlow model 'loaded'.")

    # Placeholder for actual detection function
    def detect_objects(image):
        # Simulate detection
        time.sleep(0.1) # Simulate inference time
        # In a real scenario, you'd draw bounding boxes here or pass them separately
        # For simplicity, we just pass the frame through
        # Example: image_with_detections, detections = run_inference(detection_model, image)
        return image, [{"box_coords": (50, 50, 150, 150), "label": "Object", "score": 0.8}]


    while not stop_event.is_set():
        try:
            frame_to_process = frame_queue.get(timeout=1)
            frame_queue.task_done() # Signal that the item was retrieved

            # Perform detection
            processed_image, detections = detect_objects(np.copy(frame_to_process)) # Pass a copy

            # Draw detections (optional, can be done in main thread too)
            for detection in detections:
                if detection: # If any detections were made
                    x1, y1, x2, y2 = detection["box_coords"]
                    label = detection["label"]
                    score = detection["score"]
                    cv2.rectangle(processed_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(processed_image, f"{label}: {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            try:
                processed_frame_queue.put(processed_image, timeout=1)
            except queue.Full:
                # print("Processed frame queue is full, dropping frame.")
                pass # Or handle differently

        except queue.Empty:
            continue # Wait for a new frame
        except Exception as e:
            print(f"Error in object detection thread: {e}")
            break
    print("Object detection thread finished.")


# --- Main Thread for Display ---
if __name__ == "__main__":
    print("Starting main application...")

    # Start the threads
    capture_thread = threading.Thread(target=frame_capture_thread_func,
                                      args=(VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT))
    detection_thread = threading.Thread(target=object_detection_thread_func)

    capture_thread.start()
    detection_thread.start()

    last_fps_time = time.time()
    frame_count = 0
    display_fps = 0

    try:
        while not stop_event.is_set():
            try:
                display_frame = processed_frame_queue.get(timeout=0.1) # Short timeout
                processed_frame_queue.task_done()

                # Calculate and display FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    display_fps = frame_count / (current_time - last_fps_time)
                    frame_count = 0
                    last_fps_time = current_time

                cv2.putText(display_frame, f"FPS: {display_fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow("Object Detection", display_frame)

            except queue.Empty:
                # No new frame to display, continue to process waitKey
                # You might want to display a static "waiting" image or the last known frame
                # For now, we just pass
                pass
            except Exception as e:
                print(f"Error in display loop: {e}")
                stop_event.set() # Signal threads to stop on display error
                break


            key = cv2.waitKey(1) & 0xFF # Crucial for imshow to work and to get key presses
            if key == ord('q') or key == 27: # 'q' or ESC
                print("Quit signal received.")
                stop_event.set()
                break
    finally:
        print("Cleaning up...")
        if not stop_event.is_set(): # Ensure it's set if loop exited for other reasons
            stop_event.set()

        # Wait for threads to finish
        if capture_thread.is_alive():
            print("Waiting for capture thread to join...")
            capture_thread.join(timeout=5)
        if detection_thread.is_alive():
            print("Waiting for detection thread to join...")
            detection_thread.join(timeout=5) # Add timeout for safety

        cv2.destroyAllWindows()
        print("Application finished.")