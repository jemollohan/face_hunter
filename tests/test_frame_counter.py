import pytest
from face import frame_counter

def test_initialization(frame_counter):
    assert frame_counter

def test_increment(frame_counter):
    count1 = frame_counter.get_count()
    frame_counter.increment()
    count2 = frame_counter.get_count()

    assert( (count1 + 1) == count2)

def test_reset(frame_counter):
    assert(frame_counter.get_count() == 0)

    frame_counter.increment()
    frame_counter.increment()
    assert(frame_counter.get_count() == 2)

    frame_counter.reset()
    assert(frame_counter.get_count() == 0)
