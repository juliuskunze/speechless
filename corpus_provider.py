import tarfile
import urllib.request
from pathlib import Path
from tarfile import *
from typing import List

from labeled_example import LabeledExample

test_clean = "test-clean"
tar_gz_extension = ".tar.gz"


class CorpusProvider:
    def __init__(self, base_directory: Path, base_url: str = "http://www.openslr.org/resources/12/"):
        self.base_directory = base_directory
        self.base_url = base_url
        base_directory.mkdir(exist_ok=True)

        result_directory = self._download_and_unpack_if_not_yet_done(file_name_without_extension=test_clean)
        self.corpus_directory = Path(result_directory, test_clean)
        self.examples = self.get_examples()

    def get_examples(self) -> List[LabeledExample]:
        files = [file
                 for dir in self.corpus_directory.iterdir() if dir.is_dir()
                 for subdir in dir.iterdir() if subdir.is_dir()
                 for file in subdir.iterdir() if file.is_file()]
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

        return [example(file) for file in flac_files]

    def _download_and_unpack_if_not_yet_done(self, file_name_without_extension: str) -> Path:
        file_name = file_name_without_extension + tar_gz_extension
        url = self.base_url + file_name

        tar_file = self._download_if_not_yet_done(url, Path(self.base_directory, file_name))

        return self._unpack_tar_if_not_yet_done(tar_file,
                                                target_directory=Path(self.base_directory, file_name_without_extension))

    def _unpack_tar_if_not_yet_done(self, tar_file: Path, target_directory: Path) -> Path:
        if not target_directory.is_dir():
            with tarfile.open(str(tar_file), 'r:gz') as tar:
                tar.extractall(str(target_directory), members=self._tar_members(tar))

        return target_directory

    @staticmethod
    def _tar_members(tar: TarFile, root_directory_name_to_skip="LibriSpeech/") -> List[TarInfo]:
        members = tar.getmembers()
        for member in members:
            member.name = member.name.replace(root_directory_name_to_skip, '')
        return members

    @staticmethod
    def _download_if_not_yet_done(source_url, target_path: Path) -> Path:
        if not target_path.is_file():
            urllib.request.urlretrieve(source_url, target_path)

        return target_path
