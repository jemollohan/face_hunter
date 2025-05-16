
import time


class FrameCounter:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()
    
    def increment(self):
        self.frame_count += 1
    
    def get_count(self):
        return self.frame_count
    
    def get_fps(self):
      elapsed_time = time.time() - self.start_time
      if elapsed_time > 0:
          return round(self.frame_count / elapsed_time, 2)
      return 0
    
    def reset(self):
        self.frame_count = 0
        self.start_time = time.time()