import cv2

from factory.logger_factory import get_logger

class VideoStreamer():
    def __init__(self, source=0):
        self._capture = None
        self._source = source
        self._publish_callback = []
        self._preprocess_callback = []
        self._logger = get_logger("video_streamer")

    def get_video_input(self): # 0 for webcam, or provide a path to a video file
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            self._logger.error(f"Error: Could not open video source: {self._source}")
            return None
        return cap
        
    def process(self):
        if not self._capture:
            self._capture = self.get_video_input()
            self._logger.debug("Capture has been initalized")
        
        if self._capture:
            ret, frame = self._capture.read()
            frame = self._preprocess(frame)
            if ret:
                self._publish(frame)
    
    def _publish(self, frame):
        if self._publish_callback:
            for cb in self._publish_callback:
                cb(frame)
    
    def _preprocess(self, frame):
        if self._preprocess_callback:
            for cb in self._preprocess_callback:
                temp_frame = cb(frame)
                if temp_frame:
                    frame = temp_frame
        return frame

    def add_publisher(self, callback):
        self._publish_callback.append(callback)
    
    def add_preprocess(self, fcn):
        self._preprocess_callback.append(fcn)

    
    def release(self):
        self._capture.release()


if __name__ == '__main__':
    cap = VideoStreamer()
    cap.process()
    

    
    def test_show(frame):
        cv2.imshow("Test", frame)
        cv2.waitKey(1)

    cap.add_publisher(test_show)
    for i in range(100):
        cap.process()
        cv2.waitKey(10)

    cap.release()
    cv2.destroyAllWindows