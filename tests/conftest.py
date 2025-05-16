import pytest
from face.frame_counter import FrameCounter

@pytest.fixture
def frame_counter():
    return FrameCounter()

