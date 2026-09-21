from pathlib import Path

from filelock import FileLock, Timeout

from core.errors import LibraryBusyError

BUSY_MESSAGE = (
    "The document library is busy with another operation. "
    "Wait for it to finish, then use Refresh indexing status. "
    "Do not start a second processing job or delete the lock file."
)


class LibraryLock:
    def __init__(self, data_dir: Path):
        directory = data_dir / "metadata"
        directory.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(directory / "library.lock"), timeout=0)

    def __enter__(self) -> "LibraryLock":
        try:
            self._lock.acquire()
        except Timeout as exc:
            raise LibraryBusyError(BUSY_MESSAGE) from exc
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._lock.release()


def library_is_busy(data_dir: Path) -> bool:
    try:
        with LibraryLock(data_dir):
            return False
    except LibraryBusyError:
        return True
