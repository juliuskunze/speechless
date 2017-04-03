import json
import re
from pathlib import Path

from typing import Iterable, Dict, Callable

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
    "~",  # continuation (word truncated at beginning or end of audio file)
    "#garbage#",
    "<a>",  # only occures once
    "<uhm>",
    "<uh>",
    "<hes>"
]


class UmlautDecoder:
    quote_before_umlaut = lambda text: text. \
        replace('\\"a', 'ä').replace('\\"o', 'ö').replace('\\"u', 'ü').replace('\\"s', 'ß'). \
        replace('"a', 'ä').replace('"o', 'ö').replace('"u', 'ü').replace('"s', 'ß')
    quote_after_umlaut = lambda text: text. \
        replace('a\\"', 'ä').replace('o\\"', 'ö').replace('u\\"', 'ü').replace('s\\"', 'ß'). \
        replace('a"', 'ä').replace('o"', 'ö').replace('u"', 'ü').replace('s"', 'ß')
    try_quote_before_umlaut_then_after = lambda text: UmlautDecoder.quote_after_umlaut(
        UmlautDecoder.quote_before_umlaut(text))


class CorpusDefinition:
    def __init__(self,
                 name: str,
                 umlaut_decoder: Callable[[str], str] = UmlautDecoder.quote_before_umlaut,
                 id_filter_regex=re.compile('[\s\S]*')):
        self.id_filter_regex = id_filter_regex
        self.umlaut_decoder = umlaut_decoder
        self.name = name


vm_id_German_filter_regex = re.compile("g[\s\S]*|m[\s\S]*_GER")
german_corpus_definitions_sorted_by_size = [
    CorpusDefinition("all.SC1.3.cmdi.15010.1490631864", umlaut_decoder=UmlautDecoder.quote_after_umlaut),
    CorpusDefinition("all.PD2.4.cmdi.16693.1490681127"),
    CorpusDefinition("all.SC2.3.cmdi.13887.1490631070"),
    CorpusDefinition("all.ZIPTEL.3.cmdi.63058.1490624016"),
    CorpusDefinition("all.SC10.4.cmdi.13781.1490631055",
                     umlaut_decoder=UmlautDecoder.try_quote_before_umlaut_then_after),
    CorpusDefinition("all.HEMPEL.4.cmdi.11610.1490680796"),
    CorpusDefinition("all.WaSeP.1.cmdi.21704.1490682398"),
    CorpusDefinition("all.aGender.1.cmdi.17072.1490632949"),  # contains only ".raw" files
    CorpusDefinition("all.VMEmo.1.cmdi.7826.1490627109"),  # contains only ".deo", ".nis", and ".tra" files
    CorpusDefinition("all.BROTHERS.2.cmdi.23213.1490683025"),
    CorpusDefinition("all.PD1.3.cmdi.16312.1490681066"),
    CorpusDefinition("all.VM1.3.cmdi.1508.1490625070", id_filter_regex=vm_id_German_filter_regex),
    CorpusDefinition("all.RVG-J.1.cmdi.18181.1490681704"),  # contains only ".par" and ".wav" files
    CorpusDefinition("all.HOESI.2.cmdi.15856.1490680893"),
    CorpusDefinition("all.RVG1_CLARIN.2.cmdi.19707.1490681833"),
    # contains only ".trl", ".nis", ".par" and ".txt" files
    CorpusDefinition("all.ALC.4.cmdi.16602.1490632862"),
    CorpusDefinition("all.VM2.3.cmdi.4260.1490625316", id_filter_regex=vm_id_German_filter_regex)
]


class GermanCorpusProvider(CorpusProvider):
    """
    Parses the labeled German speech data downloadable from https://clarin.phonetik.uni-muenchen.de/BASRepository/.
    """

    def __init__(self, base_directory: Path, corpus_definition: CorpusDefinition, mel_frequency_count: int = 128):
        self.corpus_definition = corpus_definition
        super().__init__(base_directory=base_directory,
                         base_source_url_or_directory="ketos:/data/mlcog/german_speech/",
                         corpus_names=[corpus_definition.name],
                         tar_gz_extension=".tgz",
                         root_compressed_directory_name_to_skip=None,
                         subdirectory_depth=1,
                         allowed_characters=frequent_characters_in_german,
                         tags_to_ignore=_tags_to_ignore,
                         id_filter_regex=corpus_definition.id_filter_regex,
                         mel_frequency_count=mel_frequency_count)

    def _extract_label_from_par(self, par_file: Path) -> str:
        par_text = read_text(par_file, encoding='utf8')

        return self._decode_german(" ".join([line[7:] for line in par_text.splitlines() if line.startswith("ORT")]))

    def _extract_labels_by_id(self, files: Iterable[Path]) -> Dict[str, str]:
        json_ending = "_annot.json"
        json_annotation_files = \
            [file for file in files if file.name.endswith(json_ending) if
             self.id_filter_regex.match(file.name[:-len(json_ending)])]

        json_extracted = dict(
            (file.name[:-len(json_ending)], self._extract_label_from_json(file)) for
            file
            in json_annotation_files)

        # TODO decide whether to parse .par files
        # par_annotation_files = [file for file in files if file.name.endswith(".par")]
        # par_extracted = dict(
        #     (name_without_extension(file), self._extract_label_from_par(file)) for file
        #     in par_annotation_files)
        # for key in set(par_extracted.keys()).intersection(set(json_extracted.keys())):
        #     if par_extracted[key] != json_extracted[key]:
        #         print()
        #
        # json_extracted.update(par_extracted)

        return json_extracted

    def _extract_label_from_json(self, json_file: Path) -> str:
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

            return self._decode_german(" ".join(words))
        except Exception:
            raise ParsingException("Error parsing annotation {}: {}".format(json_file, json_text[:500]))

    def _decode_german(self, text: str) -> str:
        # replace('é', 'e') because of TODO
        # replace('xe4', 'ä') because of F09S1MP-Mikro_Prompt_20 (+7 more): " timo hat b  xe4ten gesagt"
        # replace('.', ' ') because of all.ALC.4.cmdi.16602.1490632862: 5204018034_h_00 contains "in l.a."
        decoded = self.corpus_definition.umlaut_decoder(
            text.lower().replace('é', 'e').replace('xe4', 'ä').replace('.', ' '))

        return decoded
