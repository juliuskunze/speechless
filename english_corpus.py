import random
import re
import subprocess
import tarfile
from functools import reduce
from pathlib import Path
from tarfile import TarFile, TarInfo

from collections import Counter, OrderedDict
from lazy import lazy
from typing import Iterable, Optional, List, Callable, Tuple, Dict
from urllib import request

from corpus import Corpus, TrainingTestSplit
from grapheme_enconding import frequent_characters_in_english
from labeled_example import LabeledExample, PositionalLabel
from tools import mkdir, name_without_extension, count_summary, distinct, extension


class LibriSpeechCorpus(Corpus):
    def __init__(self, base_directory: Path,
                 base_source_url_or_directory: str = "http://www.openslr.org/resources/12/",
                 corpus_names: Iterable[str] = ("dev-clean", "dev-other", "test-clean", "test-other",
                                                "train-clean-100", "train-clean-360", "train-other-500"),
                 tar_gz_extension: str = ".tar.gz",
                 mel_frequency_count: int = 128,
                 root_compressed_directory_name_to_skip: Optional[str] = "LibriSpeech/",
                 subdirectory_depth: int = 3,
                 allowed_characters: List[chr] = frequent_characters_in_english,
                 tags_to_ignore: Iterable[str] = list(),
                 id_filter_regex=re.compile('[\s\S]*'),
                 training_test_split: Callable[[List[LabeledExample]], Tuple[
                     List[LabeledExample], List[LabeledExample]]] = TrainingTestSplit.randomly()):
        self.training_test_split = training_test_split
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

        self.files = [file
                      for directory in directories
                      for file in directory.iterdir() if file.is_file()]

        self.unfiltered_audio_files = [file for file in self.files if
                                       (file.name.lower().endswith(".flac") or file.name.lower().endswith(".wav"))]
        audio_files = [file for file in self.unfiltered_audio_files if
                       self.id_filter_regex.match(name_without_extension(file))]
        self.filtered_out_count = len(self.unfiltered_audio_files) - len(audio_files)

        positional_label_by_id = self._extract_positional_label_by_id(self.files)
        found_audio_ids = set(name_without_extension(f) for f in audio_files)
        found_label_ids = positional_label_by_id.keys()
        self.audio_ids_without_label = list(found_audio_ids - found_label_ids)
        self.label_ids_without_audio = list(found_label_ids - found_audio_ids)

        def example(audio_file: Path) -> LabeledExample:
            id = name_without_extension(audio_file)

            def correct_whitespace(text: str) -> str:
                return " ".join(text.split()).strip()

            def correct(label: str) -> str:
                return correct_whitespace(self._remove_tags_to_ignore(label))

            original_positional_label = positional_label_by_id[id]
            positional_label = original_positional_label.with_corrected_words(correct)

            return LabeledExample(audio_file,
                                  mel_frequency_count=self.mel_frequency_count,
                                  label=positional_label.label,
                                  original_label=original_positional_label.label,
                                  positional_label=positional_label)

        self.examples_with_empty = [example(file) for file in audio_files if
                                    name_without_extension(file) in positional_label_by_id.keys()]

        examples = sorted([e for e in self.examples_with_empty if e.label], key=lambda x: x.id)

        training_examples, test_examples = self.training_test_split(examples)

        super().__init__(examples=examples, training_examples=training_examples, test_examples=test_examples)

    def _remove_tags_to_ignore(self, text: str) -> str:
        return reduce(lambda text, tag: text.replace(tag, ""), self.tags_to_ignore, text)

    def _download_and_unpack_if_not_yet_done(self, corpus_name: str) -> Path:
        file_name = corpus_name + self.tar_gz_extension
        file_url_or_path = self.base_url_or_directory + file_name

        target_directory = self.base_directory / corpus_name

        if not target_directory.exists():
            tar_file = self._download_if_not_yet_done(file_url_or_path, self.base_directory / file_name)
            self._unpack_tar_if_not_yet_done(tar_file, target_directory=target_directory)

        return target_directory

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

    def _extract_positional_label_by_id(self, files: Iterable[Path]) -> Dict[str, PositionalLabel]:
        label_files = [file for file in files if file.name.endswith(".txt")]
        positional_label_by_id = OrderedDict()
        for label_file in label_files:
            with label_file.open() as f:
                for line in f.readlines():
                    parts = line.split()
                    id = parts[0]
                    label = " ".join(parts[1:])
                    positional_label_by_id[id] = (label.lower(), None)
        return positional_label_by_id

    def is_allowed(self, label: str) -> bool:
        return all(c in self.allowed_characters for c in label)

    def csv_rows(self):
        return [[" ".join(self.corpus_names),
                 self.file_type_summary,
                 len(self.unfiltered_audio_files), self.filtered_out_count, self.id_filter_regex,
                 len(self.audio_ids_without_label), str(self.audio_ids_without_label[:10]),
                 len(self.label_ids_without_audio), self.label_ids_without_audio[:10],
                 self.tag_summary,
                 len(self.examples),
                 len(self.invalid_examples_texts), self.invalid_examples_summary,
                 len(self.empty_examples), [e.id for e in self.empty_examples[:10]],
                 self.duplicate_label_count, self.most_duplicated_labels,
                 len(self.training_examples), len(self.test_examples),
                 len(self.examples_without_positional_labels)]]

    def summary(self) -> str:
        description = "File types: {}\n{}{}{}{}{}{} extracted examples, of them {} invalid, {} empty (will be excluded), {} duplicate, {} without positions.\n{} training examples, {} test examples.".format(
            self.file_type_summary,
            "Out of {} audio files, {} were excluded by regex {}\n".format(
                len(self.unfiltered_audio_files), self.filtered_out_count,
                self.id_filter_regex) if self.filtered_out_count > 0 else "",

            "{} audio files without matching label; will be excluded, e. g. {}.\n".format(
                len(self.audio_ids_without_label), self.audio_ids_without_label[:10]) if len(
                self.audio_ids_without_label) > 0 else "",

            "{} labels without matching audio file; will be excluded, e. g. {}.\n".format(
                len(self.label_ids_without_audio), self.label_ids_without_audio[:10]) if len(
                self.label_ids_without_audio) > 0 else "",

            "Removed label tags: {}\n".format(self.tag_summary) if self.tag_summary != "" else "",
            self.invalid_examples_summary,
            len(self.examples),
            len(self.invalid_examples_texts),
            len(self.empty_examples),
            self.duplicate_label_count,
            len(self.examples_without_positional_labels),
            len(self.training_examples),
            len(self.test_examples))

        return " ".join(self.corpus_names) + "\n" + "\n".join("\t" + line for line in description.splitlines())

    @lazy
    def invalid_examples_summary(self):
        return "".join([e + '\n' for e in self.invalid_examples_texts])

    @lazy
    def original_sample_rate_summary(self):
        return count_summary(self.some_original_sample_rates)

    @lazy
    def tag_summary(self):
        return count_summary(self.tags_from_all_examples)

    @lazy
    def file_type_summary(self):
        return count_summary(self.file_extensions)

    @lazy
    def invalid_examples_texts(self):
        return [
            "Invalid characters {} in {}".format(
                distinct([c for c in e.label if c not in self.allowed_characters]), str(e))
            for e in self.examples if not self.is_allowed(e.label)]

    @lazy
    def some_original_sample_rates(self):
        return [e.original_sample_rate for e in
                random.sample(self.examples, min(50, len(self.examples)))]

    @lazy
    def file_extensions(self):
        return [extension(file)
                for directory in self.corpus_directories
                for file in directory.glob('**/*.*') if file.is_file()]

    @lazy
    def empty_examples(self):
        return [e for e in self.examples_with_empty if not e.label]

    @lazy
    def duplicate_label_count(self):
        return len(self.examples) - len(set(e.label for e in self.examples))

    @lazy
    def most_duplicated_labels(self):
        return Counter([e.label for e in self.examples]).most_common(10)

    @lazy
    def tags_from_all_examples(self):
        return [counted_tag
                for e in self.examples
                for tag in self.tags_to_ignore
                for counted_tag in [tag] * e.tag_count(tag)]

    @lazy
    def examples_without_positional_labels(self):
        return [e for e in self.examples if not e.positional_label.has_positions()]
