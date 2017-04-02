import subprocess
import tarfile
from functools import reduce
from pathlib import Path
from tarfile import *

from collections import OrderedDict
from typing import List, Iterable, Optional, Dict, Callable
from urllib import request

from grapheme_enconding import frequent_characters_in_english
from labeled_example import LabeledExample
from tools import mkdir, distinct


class ParsingException(Exception):
    pass


def _extract_labels_by_id_from_txt(files: Iterable[Path]) -> Dict[str, str]:
    label_files = [file for file in files if file.name.endswith(".txt")]
    labels_by_id = dict()
    for label_file in label_files:
        with label_file.open() as f:
            for line in f.readlines():
                parts = line.split()
                id = parts[0]
                label = " ".join(parts[1:])
                labels_by_id[id] = label.lower()
    return labels_by_id


class CorpusProvider:
    def __init__(self, base_directory: Path,
                 base_source_url_or_directory: str = "http://www.openslr.org/resources/12/",
                 corpus_names: Iterable[str] = ("dev-clean", "dev-other", "test-clean", "test-other",
                                                "train-clean-100", "train-clean-360", "train-other-500"),
                 tar_gz_extension: str = ".tar.gz",
                 mel_frequency_count: int = 128,
                 root_compressed_directory_name_to_skip: Optional[str] = "LibriSpeech/",
                 subdirectory_depth: int = 2,
                 allowed_characters: List[chr] = frequent_characters_in_english,
                 tags_to_ignore: Iterable[str] = list(),
                 labels_by_id_extractor: Callable[[Iterable[Path]], Dict[str, str]] = _extract_labels_by_id_from_txt):
        self.tags_to_ignore = tags_to_ignore
        self.allowed_characters = allowed_characters
        self.labels_by_id_extractor = labels_by_id_extractor
        self.subdirectory_depth = subdirectory_depth
        self.root_compressed_directory_name_to_skip = root_compressed_directory_name_to_skip
        self.base_directory = base_directory
        self.base_url_or_directory = base_source_url_or_directory
        self.tar_gz_extension = tar_gz_extension
        self.mel_frequency_count = mel_frequency_count
        self.corpus_names = corpus_names
        mkdir(base_directory)
        self.corpus_directories = [self._download_and_unpack_if_not_yet_done(corpus_name=corpus_name) for corpus_name in
                                   corpus_names]
        self.examples = self._get_examples()
        self.examples_by_id = dict([(e.id, e) for e in self.examples])

    def _get_examples(self) -> List[LabeledExample]:
        directories = self.corpus_directories
        for i in range(self.subdirectory_depth):
            directories = [subdirectory
                           for directory in directories
                           for subdirectory in directory.iterdir() if subdirectory.is_dir()]

        files = [file
                 for directory in directories
                 for file in directory.iterdir() if file.is_file()]
        audio_files = [file for file in files if file.name.endswith(".flac") or file.name.endswith(".wav")]

        labels_with_tags_by_id = self.labels_by_id_extractor(files)
        if len(audio_files) != len(labels_with_tags_by_id):
            raise ParsingException(
                "Found {} audio files, but {} labels in corpus {}.".format(len(audio_files),
                                                                           len(labels_with_tags_by_id),
                                                                           self.corpus_names))

        def example(audio_file: Path) -> LabeledExample:
            return LabeledExample.from_file(audio_file, label_from_id=lambda id: self._remove_tags_to_ignore(
                labels_with_tags_by_id[id]),
                                            mel_frequency_count=self.mel_frequency_count,
                                            original_label_with_tags_from_id=lambda id: labels_with_tags_by_id[id])

        return sorted([example(file) for file in audio_files], key=lambda x: x.id)

    def _remove_tags_to_ignore(self, text: str) -> str:
        return reduce(lambda text, tag: text.replace(tag, ""), self.tags_to_ignore, text)

    def _download_and_unpack_if_not_yet_done(self, corpus_name: str) -> Path:
        file_name = corpus_name + self.tar_gz_extension
        file_url_or_path = self.base_url_or_directory + file_name

        target_directory = self.base_directory / corpus_name

        if not target_directory.exists():
            tar_file = self._download_if_not_yet_done(file_url_or_path, self.base_directory / file_name)
            self._unpack_tar_if_not_yet_done(tar_file, target_directory=target_directory)

        return [sub_directory for sub_directory in target_directory.iterdir() if sub_directory.is_dir()][0]

    def _unpack_tar_if_not_yet_done(self, tar_file: Path, target_directory: Path):
        if not target_directory.is_dir():
            with tarfile.open(str(tar_file), 'r:gz') as tar:
                tar.extractall(str(target_directory),
                               members=self._tar_members_root_directory_skipped_if_specified(tar))

    def _tar_members_root_directory_skipped_if_specified(self, tar: TarFile) -> List[TarInfo]:
        members = tar.getmembers()

        if self.root_compressed_directory_name_to_skip is not None:
            for member in members:
                member.name = member.name.replace(self.root_compressed_directory_name_to_skip, '')

        return members

    def _download_if_not_yet_done(self, source_path_or_url: str, target_path: Path) -> Path:
        if not target_path.is_file():
            print("Downloading corpus {} to {}".format(source_path_or_url, target_path))
            if self.base_url_or_directory.startswith("http"):
                request.urlretrieve(source_path_or_url, str(target_path))
            else:
                try:
                    subprocess.check_output(["scp", source_path_or_url, str(target_path)], stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    raise ParsingException("Copying failed: " + str(e.output))

        return target_path

    def is_allowed(self, label: str) -> bool:
        return all(c in self.allowed_characters for c in label)

    def summary(self) -> str:
        invalid_examples = [
            " Invalid characters {} in {}".format(
                distinct([c for c in x.label if c not in self.allowed_characters]), str(x))
            for x in self.examples if not self.is_allowed(x.label)]

        example_count_by_tag = OrderedDict(
            [(tag, len([example for example in self.examples if example.contains_tag(tag)])) for tag in
             self.tags_to_ignore])

        duplicate_label_count = len(self.examples) - len(set(e.label for e in self.examples))

        empty_examples = [example for example in self.examples if example.label == ""]

        return "{}:\n{} {} total with {} invalid, {} empty, {} duplicate\n Examples that had special tags: {}\n".format(
            " ".join(self.corpus_names),
            "".join([e + '\n' for e in invalid_examples]),
            len(self.examples),
            len(invalid_examples),
            len(empty_examples),
            duplicate_label_count,
            ", ".join(["{}: {}".format(tag, count) for tag, count in example_count_by_tag.items() if count != 0]))
