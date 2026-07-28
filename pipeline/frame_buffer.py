"""
Sliding window frame buffer for clip-based models (violence classifier).

Keeps the last N frames in memory so clip-based models always have
enough temporal context, without the runner needing to know how many
frames each model wants internally.
"""

from collections import deque


class FrameBuffer:
    def __init__(self, max_len: int = 32):
        self.max_len = max_len
        self._buffer = deque(maxlen=max_len)

    def push(self, frame):
        self._buffer.append(frame)

    def get_clip(self):
        """Returns current buffered frames as a list (may be shorter than max_len early on)."""
        return list(self._buffer)

    def is_ready(self, min_frames: int = 2) -> bool:
        return len(self._buffer) >= min_frames

    def __len__(self):
        return len(self._buffer)
