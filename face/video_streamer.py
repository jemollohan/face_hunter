import cv2

class VideoStreamer():
    def __init__(self, source=0):
        self._capture = None
        self._source = source
        self._publish_callabck = []
        self._preprocess_callback = []


    def get_video_input(self): # 0 for webcam, or provide a path to a video file
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            print(f"Error: Could not open video source: {self._source}")
            return None
        return cap
        
    def process(self):
        if not self._capture:
            self._capture = self.get_video_input()
        
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
