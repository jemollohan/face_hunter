from frame_counter import FrameCounter
from factory.logger_factory import get_logger
from factory.video_streamer_factory import get_local_camera
logger = get_logger


def main():
    video_cap = get_local_camera()
    counter = FrameCounter()

    logger.debug("Initializing Writer...")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
    video_writer = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480)) # Output file name, codec, frames per second, frame size

    if video_cap:
        while True:
            ret, frame = video_cap.read()
            if not ret:
                logger.error("Error: Failed to grab frame or end of video.")
                break

            counter.increment()

            person_boxes, person_scores = detect_persons(frame, model)
            frame_with_detections = draw_detections(frame.copy(), person_boxes, person_scores, counter.get_fps())
            record_video(frame, video_writer)

            cv2.imshow("Person Detection", frame_with_detections)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()