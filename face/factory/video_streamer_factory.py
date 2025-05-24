import cv2

from video_streamer import VideoStreamer


def preprocess_rgb_format(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def get_local_camera():
    cap = VideoStreamer
    cap.add_preprocess(VideoStreamer)

    return cap