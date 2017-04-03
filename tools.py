from pathlib import Path

from collections import OrderedDict, Counter
from os import makedirs, path
from typing import List, Any


def single(sequence: List) -> Any:
    first = sequence[0]

    assert (len(sequence) == 1)

    return first


def single_or_none(sequence: List) -> Any:
    return next(iter(sequence), None)


def read_text(path: Path, encoding=None):
    """
    Not Path.read_text for compatibility with Python 3.4.
    """
    with path.open(encoding=encoding) as f:
        return f.read()


def mkdir(directory: Path):
    """
    Not Path.mkdir() for compatibility with Python 3.4.
    """
    makedirs(str(directory), exist_ok=True)


def home_directory() -> Path:
    """
    Not Path.home() for compatibility with Python 3.4.
    """
    return Path(path.expanduser('~'))


def name_without_extension(audio_file: Path) -> str:
    return path.splitext(audio_file.name)[0]


def extension(audio_file: Path) -> str:
    return path.splitext(audio_file.name)[1]


def distinct(sequence: List) -> List:
    return list(OrderedDict.fromkeys(sequence))


def count_summary(list: List) -> str:
    return ", ".join(["{}: {}".format(tag, count) for tag, count in Counter(list).most_common()])
