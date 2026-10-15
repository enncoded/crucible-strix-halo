"""Private, short-lived pixel files: the engine reads image pixels from them, not from the JSON line.

A JSON line of base64 pixels costs the engine about 20 ms per MB to read, and every other batch session waits
for it. Each image goes to a mode-0600 file with a random name in a mode-0700
directory under /dev/shm (memory, not disk). The engine opens only the files of images that it encodes.
The backend removes a request's files when the engine has finished with the request, and close() removes
the directory. A new store removes the directories of dead servers of this user.
"""
import os
import secrets
import shutil
import tempfile
import threading
from pathlib import Path

PREFIX = "crucible-vision-"


def default_root():
    return Path("/dev/shm") if os.path.isdir("/dev/shm") else Path(tempfile.gettempdir())


class PixelFiles:
    """The files of one engine request. remove() is idempotent."""

    def __init__(self, paths):
        self.paths = tuple(paths)
        self._removed = False

    def remove(self):
        if self._removed:
            return
        self._removed = True
        for path in self.paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


class PixelStore:
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else default_root()
        self.directory = None
        self.lock = threading.Lock()

    def _ensure(self):
        with self.lock:
            if self.directory is None:
                _remove_stale(self.root)
                self.directory = tempfile.mkdtemp(prefix=f"{PREFIX}{os.getpid()}-", dir=self.root)
                os.chmod(self.directory, 0o700)
            return self.directory

    def write(self, images):
        """Write each image's RGB bytes to a new private file. Returns PixelFiles in image order."""
        directory = self._ensure()
        paths = []
        try:
            for image in images:
                path = os.path.join(directory, secrets.token_hex(16))
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
                paths.append(path)
                try:
                    view = memoryview(image.rgb)
                    while view:
                        view = view[os.write(fd, view):]
                finally:
                    os.close(fd)
        except BaseException:
            PixelFiles(paths).remove()
            raise
        return PixelFiles(paths)

    def close(self):
        with self.lock:
            if self.directory is not None:
                shutil.rmtree(self.directory, ignore_errors=True)
                self.directory = None


def _remove_stale(root):
    """Remove pixel directories of this user whose server process no longer exists."""
    try:
        entries = list(os.scandir(root))
    except OSError:
        return
    for entry in entries:
        if not entry.name.startswith(PREFIX):
            continue
        pid = entry.name[len(PREFIX):].split("-", 1)[0]
        try:
            if not pid.isdigit() or entry.stat(follow_symlinks=False).st_uid != os.getuid():
                continue
            if not entry.is_dir(follow_symlinks=False):
                continue
            os.kill(int(pid), 0)
        except ProcessLookupError:
            shutil.rmtree(entry.path, ignore_errors=True)
        except (OSError, ValueError):
            continue
