"""fcntl.flock stand-in for Windows, built on msvcrt byte-range locks."""
from __future__ import annotations

import msvcrt
import os
import time

LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8


def flock(fd, operation: int) -> None:
    if not isinstance(fd, int):
        fd = fd.fileno()
    os.lseek(fd, 0, os.SEEK_SET)
    if operation & LOCK_UN:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    while True:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return
        except OSError as e:
            if operation & LOCK_NB:
                raise BlockingIOError(*e.args) from e
            time.sleep(0.05)
