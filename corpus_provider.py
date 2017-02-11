import tarfile
import urllib.request
from os import makedirs
from pathlib import Path
from tarfile import *
from typing import List, Iterable

from labeled_example import LabeledExample

tar_gz_extension = ".tar.gz"


class CorpusProvider:
    def __init__(self, base_directory: Path,
                 base_url: str = "http://www.openslr.org/resources/12/",
                 corpus_names: Iterable[str] = ("dev-clean", "dev-other", "test-clean", "test-other",
                                                "train-clean-100", "train-clean-360", "train-other-500")):
        self.base_directory = base_directory
        self.base_url = base_url
        # not Path.mkdir() for compatibility with Python 3.4
        makedirs(str(base_directory), exist_ok=True)
        self.corpus_directories = [self._download_and_unpack_if_not_yet_done(corpus_name=corpus_name) for corpus_name in
                                   corpus_names]
        self.examples = self.get_examples()

    def get_examples(self) -> List[LabeledExample]:
        files = [file
                 for corpus_directory in self.corpus_directories
                 for directory in corpus_directory.iterdir() if directory.is_dir()
                 for subdirectory in directory.iterdir() if subdirectory.is_dir()
                 for file in subdirectory.iterdir() if file.is_file()]
        flac_files = [file for file in files if file.name.endswith(".flac")]
        label_files = [file for file in files if file.name.endswith(".txt")]
        labels_by_id = dict()
        for label_file in label_files:
            with label_file.open() as f:
                for line in f.readlines():
                    parts = line.split()
                    id = parts[0]
                    label = " ".join(parts[1:])
                    labels_by_id[id] = label
        assert (len(flac_files) == len(labels_by_id))

        def example(flac_file: Path) -> LabeledExample:
            id = flac_file.name.replace(".flac", "")
            return LabeledExample(id, flac_file, labels_by_id[id])

        return sorted([example(file) for file in flac_files], key=lambda x: x.id)

    def _download_and_unpack_if_not_yet_done(self, corpus_name: str) -> Path:
        file_name = corpus_name + tar_gz_extension
        url = self.base_url + file_name

        tar_file = self._download_if_not_yet_done(url, self.base_directory / file_name)

        return self._unpack_tar_if_not_yet_done(tar_file,
                                                target_directory=self.base_directory / corpus_name) / corpus_name

    def _unpack_tar_if_not_yet_done(self, tar_file: Path, target_directory: Path) -> Path:
        if not target_directory.is_dir():
            with tarfile.open(str(tar_file), 'r:gz') as tar:
                tar.extractall(str(target_directory), members=self._tar_members_top_folder_skipped(tar))

        return target_directory

    @staticmethod
    def _tar_members_top_folder_skipped(tar: TarFile, root_directory_name_to_skip="LibriSpeech/") -> List[TarInfo]:
        members = tar.getmembers()
        for member in members:
            member.name = member.name.replace(root_directory_name_to_skip, '')
        return members

    @staticmethod
    def _download_if_not_yet_done(source_url: str, target_path: Path) -> Path:
        if not target_path.is_file():
            print("Downloading corpus {} to {}".format(source_url, target_path))
            urllib.request.urlretrieve(source_url, str(target_path))

        return target_path
