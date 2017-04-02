import re
import subprocess
import tarfile
from functools import reduce
from pathlib import Path
from tarfile import *

from collections import OrderedDict
from typing import List, Iterable, Optional, Dict
from urllib import request

from grapheme_enconding import frequent_characters_in_english
from labeled_example import LabeledExample
from tools import mkdir, distinct, name_without_extension


class ParsingException(Exception):
    pass


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
                 id_filter_regex=re.compile('[\s\S]*')):
        self.id_filter_regex = id_filter_regex
        self.tags_to_ignore = tags_to_ignore
        self.allowed_characters = allowed_characters
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

        directories = self.corpus_directories
        for i in range(self.subdirectory_depth):
            directories = [subdirectory
                           for directory in directories
                           for subdirectory in directory.iterdir() if subdirectory.is_dir()]

        files = [file
                 for directory in directories
                 for file in directory.iterdir() if file.is_file()]

        self.unfiltered_audio_files = [file for file in files if
                                       (file.name.endswith(".flac") or file.name.endswith(".wav"))]
        audio_files = [file for file in self.unfiltered_audio_files if
                       self.id_filter_regex.match(name_without_extension(file))]
        self.filtered_out_count = len(self.unfiltered_audio_files) - len(audio_files)

        labels_with_tags_by_id = self._extract_labels_by_id(files)
        found_audio_ids = set(name_without_extension(f) for f in audio_files)
        found_label_ids = labels_with_tags_by_id.keys()
        self.audio_ids_without_label = list(found_audio_ids - found_label_ids)
        self.label_ids_without_audio = list(found_label_ids - found_audio_ids)

        def example(audio_file: Path) -> LabeledExample:
            return LabeledExample.from_file(audio_file, label_from_id=lambda id: self._remove_tags_to_ignore(
                labels_with_tags_by_id[id]),
                                            mel_frequency_count=self.mel_frequency_count,
                                            original_label_with_tags_from_id=lambda id: labels_with_tags_by_id[id])

        self.examples = sorted(
            [example(file) for file in audio_files if name_without_extension(file) in labels_with_tags_by_id.keys()],
            key=lambda x: x.id)
        self.examples_by_id = dict([(e.id, e) for e in self.examples])

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
                    raise IOError("Copying failed: " + str(e.output))

        return target_path

    def _extract_labels_by_id(self, files: Iterable[Path]) -> Dict[str, str]:
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

        return "{}:\n{}{}{}{} {} total with {} invalid, {} empty, {} duplicate\n Examples that had special tags: {}\n".format(
            " ".join(self.corpus_names),
            " Originally found {} audio files, {} were filtered out with regex {}\n".format(
                len(self.unfiltered_audio_files), self.filtered_out_count,
                self.id_filter_regex) if self.filtered_out_count > 0 else "",

            " Found {} audio files without match label; will be excluded, e. g. {}.\n".format(
                len(self.audio_ids_without_label), self.audio_ids_without_label[:10]) if len(
                self.audio_ids_without_label) > 0 else "",

            " Found {} labels without matching audio file; will be excluded, e. g. {}.\n".format(
                len(self.label_ids_without_audio), self.label_ids_without_audio[:10]) if len(
                self.label_ids_without_audio) > 0 else "",

            "".join([e + '\n' for e in invalid_examples]),
            len(self.examples),
            len(invalid_examples),
            len(empty_examples),
            duplicate_label_count,
            ", ".join(["{}: {}".format(tag, count) for tag, count in example_count_by_tag.items() if count != 0]))
