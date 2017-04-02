import json
from pathlib import Path

from typing import List, Iterable, Dict

from corpus_provider import CorpusProvider, ParsingException
from grapheme_enconding import frequent_characters_in_german
from tools import read_text, single, single_or_none

_tags_to_ignore = [
    "<usb>",  # truncated in the beginning
    "<häs>",  # "äh", "ähm" etc.
    "<%>",  # slip of the tongue, voice without meaning
    "*",  # slip of the tongue, following word unclear, but still comprehensible
    "<äh>",
    "<ähm>",
    "<hm>",
    "$",  # indicates that the following character is pronounced in isolation, e. g. $u-$s-$a
    "~"  # continuation (word truncated at beginning or end of audio file)
]

german_corpus_names_sorted_by_size = [
    "all.SC1.3.cmdi.15010.1490631864",
    "all.PD2.4.cmdi.16693.1490681127",
    "all.SC2.3.cmdi.13887.1490631070",
    "all.ZIPTEL.3.cmdi.63058.1490624016",
    "all.SC10.4.cmdi.13781.1490631055",
    "all.HEMPEL.4.cmdi.11610.1490680796",
    "all.WaSeP.1.cmdi.21704.1490682398",
    "all.aGender.1.cmdi.17072.1490632949",
    "all.VMEmo.1.cmdi.7826.1490627109",
    "all.BROTHERS.2.cmdi.23213.1490683025",
    "all.PD1.3.cmdi.16312.1490681066",
    "all.VM1.3.cmdi.1508.1490625070",
    "all.RVG-J.1.cmdi.18181.1490681704",
    "all.HOESI.2.cmdi.15856.1490680893",
    "all.RVG1_CLARIN.2.cmdi.19707.1490681833",
    "all.ALC.4.cmdi.16602.1490632862",
    "all.VM2.3.cmdi.4260.1490625316"
]


def german_corpus_provider(base_directory: Path,
                           corpus_names: List[str] = german_corpus_names_sorted_by_size) -> CorpusProvider:
    return CorpusProvider(base_directory=base_directory,
                          base_source_url_or_directory="ketos:/data/mlcog/german_speech/",
                          corpus_names=corpus_names,
                          tar_gz_extension=".tgz",
                          root_compressed_directory_name_to_skip=None,
                          subdirectory_depth=1,
                          allowed_characters=frequent_characters_in_german,
                          tags_to_ignore=_tags_to_ignore,
                          labels_by_id_extractor=_extract_labels_by_id_from_jsons)


def _extract_labels_by_id_from_jsons(files: Iterable[Path]) -> Dict[str, str]:
    ending = "_annot.json"
    json_annotation_files = [file for file in files if file.name.endswith(ending)]

    return dict(
        (file.name[:-len(ending)], _extract_label_from_german_corpus(file)) for file
        in json_annotation_files)


def _extract_label_from_german_corpus(json_file: Path) -> str:
    json_text = read_text(json_file, encoding='utf8')
    label_names = ("ORT", "word")
    try:
        j = json.loads(json_text)
        levels = j["levels"]

        def is_level_empty(level: json) -> bool:
            return len(level["items"]) == 0

        def is_level_useful(level: json) -> bool:
            if is_level_empty(level):
                return False

            return any([label for label in level["items"][0]["labels"] if label["name"] in label_names])

        def word(transcription: json) -> str:
            labels = transcription["labels"]

            matching_labels = [label for label in labels if label["name"] in label_names]

            if len(matching_labels) == 0:
                raise Exception("No matching label names, found {} instead.".format(
                    [label["name"] for label in labels]))

            matching_label = single(matching_labels)
            return matching_label["value"]

        has_empty_levels = len([level for level in levels if is_level_empty(level)]) != 0

        words = single_or_none([[word(transcription) for
                                 transcription in level["items"]] for level in levels if is_level_useful(level)])

        if words is None and has_empty_levels:
            return ""

        return _decode_german(" ".join(words))
    except Exception:
        raise ParsingException("Error parsing annotation {}: {}".format(json_file, json_text[:500]))


def _decode_german(text: str) -> str:
    decoded = text.lower(). \
        replace("\"a", "ä").replace("\"o", "ö").replace("\"u", "ü"). \
        replace("a\"", "ä").replace("o\"", "ö").replace("u\"", "ü"). \
        replace("\"s", "ß").replace("é", "e")

    if "aü" in decoded:
        print("aü detected: " + decoded)

    return decoded
