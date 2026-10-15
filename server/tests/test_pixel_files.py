"""Image pixels go to the engine as private files, not in the JSON line."""
import os
import stat
import subprocess
import sys

import pytest

from server.pixel_files import PixelStore
from server.vision import VisionImage, image_digest


def image(height=32, width=64, fill=7):
    rgb = bytes([fill]) * (height * width * 3)
    return VisionImage(height, width, rgb, image_digest(height, width, rgb))


def test_files_are_private_and_hold_the_pixels(tmp_path):
    store = PixelStore(root=tmp_path)
    one, two = image(fill=1), image(fill=2)
    files = store.write([one, two])
    assert len(files.paths) == 2 and len(set(files.paths)) == 2
    directory = os.path.dirname(files.paths[0])
    assert stat.S_IMODE(os.stat(directory).st_mode) == 0o700
    assert os.path.basename(directory).startswith(f"crucible-vision-{os.getpid()}-")
    for path, item in zip(files.paths, (one, two)):
        assert os.path.isabs(path)
        assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
        assert open(path, "rb").read() == item.rgb
        # Names carry no image identity.
        assert item.digest[:16] not in path
    files.remove()
    assert not any(os.path.exists(path) for path in files.paths)
    files.remove()  # idempotent
    store.close()
    assert not os.path.exists(directory)


def test_close_removes_files_that_a_request_still_holds(tmp_path):
    store = PixelStore(root=tmp_path)
    files = store.write([image()])
    store.close()
    assert not os.path.exists(files.paths[0])
    files.remove()  # no error after close


def test_a_failed_write_leaves_no_file(tmp_path, monkeypatch):
    store = PixelStore(root=tmp_path)
    calls = []
    real = os.write

    def failing(fd, data):
        calls.append(fd)
        if len(calls) == 2:
            raise OSError(28, "No space left on device")
        return real(fd, data)
    monkeypatch.setattr(os, "write", failing)
    with pytest.raises(OSError):
        store.write([image(fill=1), image(fill=2)])
    monkeypatch.undo()
    directory = store.directory
    assert directory and os.listdir(directory) == []
    store.close()


def test_stale_directories_of_dead_servers_are_removed(tmp_path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    stale = tmp_path / f"crucible-vision-{dead.pid}-abc"
    stale.mkdir(mode=0o700)
    (stale / "left").write_bytes(b"x")
    live = tmp_path / f"crucible-vision-{os.getppid()}-def"
    live.mkdir(mode=0o700)
    other = tmp_path / "unrelated"
    other.mkdir()
    store = PixelStore(root=tmp_path)
    store.write([image()]).remove()
    assert not stale.exists() and live.exists() and other.exists()
    store.close()


def test_the_default_root_is_memory_backed():
    store = PixelStore()
    assert str(store.root) in ("/dev/shm",) or not os.path.isdir("/dev/shm")
