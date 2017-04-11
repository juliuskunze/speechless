from itertools import groupby
from pathlib import Path
from time import strftime

from collections import OrderedDict, Counter
from os import makedirs, path
from typing import List, Any, Iterable


def single(sequence: List) -> Any:
    first = sequence[0]

    assert (len(sequence) == 1)

    return first


def single_or_none(sequence: List) -> Any:
    assert (len(sequence) <= 1)

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


def count_summary(sequence: List) -> str:
    return ", ".join(["{}: {}".format(tag, count) for tag, count in Counter(sequence).most_common()])


def group(iterable, key, value=lambda x: x):
    return dict((k, list(map(value, values))) for k, values in groupby(sorted(iterable, key=key), key))


def timestamp() -> str:
    return strftime("%Y%m%d-%H%M%S")


def duplicates(sequence: Iterable) -> List:
    return [item for item, count in Counter(sequence).items() if count > 1]
