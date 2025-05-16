import pytest
import time
from unittest import mock

from face import frame_counter

def test_initialization(frame_counter):
    assert frame_counter

def test_increment(frame_counter):
    count1 = frame_counter.get_count()
    frame_counter.increment()
    count2 = frame_counter.get_count()

    assert( (count1 + 1) == count2)

def test_reset(frame_counter):
    time_stamp_value = 1678886400.0
    with mock.patch('time.time') as mock_time:
        mock_time.return_value = time_stamp_value
        assert(frame_counter.get_count() == 0)

        frame_counter.increment()
        frame_counter.increment()
        assert(frame_counter.get_count() == 2)

        frame_counter.reset()
        assert(frame_counter.get_count() == 0)
        assert(pytest.approx(frame_counter.start_time) == time_stamp_value)


def test_fps(frame_counter):
    time_stamp_value = 1678886400.0
    with mock.patch('time.time') as mock_time:
        mock_time.return_value = time_stamp_value
        frame_counter.reset()
        frame_counter.increment()
        frame_counter.increment()
        mock_time.return_value = time_stamp_value + 2
        assert(pytest.approx(frame_counter.get_fps()) == 1)

