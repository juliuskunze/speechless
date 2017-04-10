import json
import re
from pathlib import Path

from typing import Iterable, Dict, Callable, Optional, List, Tuple
from xml.etree import ElementTree

from corpus import ParsingException, TrainingTestSplit, CombinedCorpus
from english_corpus import LibriSpeechCorpus
from grapheme_enconding import frequent_characters_in_german
from labeled_example import LabeledExample
from tools import read_text, single, single_or_none, name_without_extension

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
    "<hes>",
    "/"  # in few examples of ALC corpus
]


class UmlautDecoder:
    none = lambda text: text
    quote_before_umlaut = lambda text: text. \
        replace('\\"a', 'ä').replace('\\"o', 'ö').replace('\\"u', 'ü').replace('\\"s', 'ß'). \
        replace('"a', 'ä').replace('"o', 'ö').replace('"u', 'ü').replace('"s', 'ß')
    quote_after_umlaut = lambda text: text. \
        replace('a\\"', 'ä').replace('o\\"', 'ö').replace('u\\"', 'ü').replace('s\\"', 'ß'). \
        replace('a"', 'ä').replace('o"', 'ö').replace('u"', 'ü').replace('s"', 'ß')
    try_quote_before_umlaut_then_after = lambda text: UmlautDecoder.quote_after_umlaut(
        UmlautDecoder.quote_before_umlaut(text))


class GermanClarinCorpus(LibriSpeechCorpus):
    """
    Parses the labeled German speech data downloadable from https://clarin.phonetik.uni-muenchen.de/BASRepository/.
    """

    def __init__(self,
                 corpus_name: str,
                 base_directory: Path,
                 base_source_url_or_directory: str = "ketos:/projects/korpora/speech/",
                 umlaut_decoder: Callable[[str], str] = UmlautDecoder.quote_before_umlaut,
                 tar_gz_extension: str = ".tgz",
                 mel_frequency_count: int = 128,
                 root_compressed_directory_name_to_skip: Optional[str] = None,
                 subdirectory_depth: int = 2,
                 tags_to_ignore: Iterable[str] = _tags_to_ignore,
                 id_filter_regex=re.compile('[\s\S]*'),
                 training_test_split: Callable[[List[LabeledExample]], Tuple[
                     List[LabeledExample], List[LabeledExample]]] = TrainingTestSplit.randomly_by_directory()):
        self.umlaut_decoder = umlaut_decoder

        super().__init__(base_directory=base_directory,
                         base_source_url_or_directory=base_source_url_or_directory,
                         corpus_names=[corpus_name],
                         tar_gz_extension=tar_gz_extension,
                         root_compressed_directory_name_to_skip=root_compressed_directory_name_to_skip,
                         subdirectory_depth=subdirectory_depth,
                         allowed_characters=frequent_characters_in_german,
                         tags_to_ignore=tags_to_ignore,
                         id_filter_regex=id_filter_regex,
                         mel_frequency_count=mel_frequency_count,
                         training_test_split=training_test_split)

    def _extract_label_from_par(self, par_file: Path) -> str:
        par_text = read_text(par_file, encoding='utf8')

        return self._decode_german(
            " ".join([line.split("\t")[-1] for line in par_text.splitlines() if line.startswith("ORT")]))

    def _extract_labels_by_id(self, files: Iterable[Path]) -> Dict[str, str]:
        json_ending = "_annot.json"
        json_annotation_files = \
            [file for file in files if file.name.endswith(json_ending) if
             self.id_filter_regex.match(file.name[:-len(json_ending)])]

        json_extracted = dict(
            (file.name[:-len(json_ending)], self._extract_label_from_json(file)) for
            file
            in json_annotation_files)

        par_annotation_files = [file for file in files if file.name.lower().endswith(".par")]

        par_extracted = dict(
            (name_without_extension(file), self._extract_label_from_par(file)) for file in par_annotation_files)
        for key in set(par_extracted.keys()).intersection(set(json_extracted.keys())):
            if par_extracted[key] != json_extracted[key]:
                print('{}: par label "{}" differs from json label "{}"'.format(key, par_extracted[key],
                                                                               json_extracted[key]))

        json_extracted.update(par_extracted)

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
        # replace('.', ' ') because of ALC: 5204018034_h_00 contains "in l.a."
        # replace('-', ' ') because of some examples in e. g. ZIPTEL, PD2, SC10 like the following:
        # SC10: awed5070: "darf ich eine ic-fahrt zwischendurch unterbrechen"
        decoded = self.umlaut_decoder(
            text.lower().replace('é', 'e').replace('xe4', 'ä').replace('.', ' ').replace('-', ' '))

        return decoded


# from the VM1 readme:
# K = German, same room, no push button
# L = German, separated room, no push button
# M = Geramn, separated room, push button
# N = German, same room, push button
# G = German, separated room, push button
# Q = same as M but 'Denglisch'
# (Germans speaking English)
# R = same as N but American English
# (recording site 'C') or
# 'Denglish' (recording site 'K')
# Z = German, test recording in the
# szenario 'travel planning'
# by TP 13 Hamburg
# J = same as G but extended scenario of
# 1995, 1996
# S = same as M but mixed German-English
# W = same as M but with a Wizard
# Y = Japanese, same room, push button
vm1_id_german_filter_regex = re.compile("[klmngzjw][\s\S]*")

# from the VM2 readme:
# g(erman), e(nglish), j(apanese), m(ultilingual), n(oise)
vm2_id_german_filter_regex = re.compile("g[\s\S]*|m[\s\S]*_GER")


def rvg_j(base_directory):
    return GermanClarinCorpus("all.RVG-J.1.cmdi.18181.1490681704", base_directory)


def clarin_corpora_sorted_by_size(base_directory: Path) -> List[GermanClarinCorpus]:
    return [
        GermanClarinCorpus("all.SC1.3.cmdi.15010.1490631864", base_directory,
                           umlaut_decoder=UmlautDecoder.quote_after_umlaut),
        GermanClarinCorpus("all.PD2.4.cmdi.16693.1490681127", base_directory),
        GermanClarinCorpus("all.ZIPTEL.3.cmdi.63058.1490624016", base_directory),
        GermanClarinCorpus("all.SC10.4.cmdi.13781.1490631055", base_directory,
                           umlaut_decoder=UmlautDecoder.try_quote_before_umlaut_then_after),
        GermanClarinCorpus("all.HEMPEL.4.cmdi.11610.1490680796", base_directory),
        GermanClarinCorpus("all.PD1.3.cmdi.16312.1490681066", base_directory),
        GermanClarinCorpus("all.VM1.3.cmdi.1508.1490625070", base_directory,
                           id_filter_regex=vm1_id_german_filter_regex,
                           training_test_split=TrainingTestSplit.training_only),
        rvg_j(base_directory),
        GermanClarinCorpus("all.ALC.4.cmdi.16602.1490632862", base_directory,
                           training_test_split=TrainingTestSplit.training_only),
        GermanClarinCorpus("all.VM2.3.cmdi.4260.1490625316", base_directory,
                           id_filter_regex=vm2_id_german_filter_regex,
                           training_test_split=TrainingTestSplit.training_only)
    ]


class GermanVoxforgeCorpus(GermanClarinCorpus):
    def __init__(self, base_directory: Path):
        super().__init__(
            corpus_name="german-speechdata-package-v2",
            base_directory=base_directory,
            base_source_url_or_directory="http://www.repository.voxforge1.org/downloads/de/",
            tar_gz_extension=".tar.gz",
            subdirectory_depth=1,
            umlaut_decoder=UmlautDecoder.none,
            # exclude files starting with dot:
            id_filter_regex=re.compile('[^.][\s\S]*', ),
            training_test_split=TrainingTestSplit.by_directory())

    def _extract_labels_by_id(self, files: Iterable[Path]):
        xml_ending = ".xml"

        microphone_endings = [
            "_Yamaha",
            "_Kinect-Beam",
            "_Kinect-RAW",
            "_Realtek",
            "_Samson",
            "_Microsoft-Kinect-Raw"
        ]

        xml_files = [file for file in files if file.name.endswith(xml_ending) if
                     self.id_filter_regex.match(name_without_extension(file))]

        return dict(
            (name_without_extension(file) + microphone_ending, self._extract_label_from_xml(file))
            for file in xml_files
            for microphone_ending in microphone_endings
            if (Path(file.parent) / (name_without_extension(file) + microphone_ending + ".wav")).exists())

    def _decode_german(self, text: str) -> str:
        # replace("co2", "co zwei") for e. g. 2014-03-19-16-39-20_Kinect-Beam
        # replace('ț', 't') for e. g. 2015-01-27-11-32-50_Kinect-Beam:
        # "durchlaufende wagen bis constanța wie sie vor dem krieg existierten wurden allerdings nicht mehr eingeführt"
        # replace('š', 's') for e. g. 2015-01-27-13-33-01_Kinect-Beam
        # replace('č', 'c') for e. g. 2015-01-28-11-49-53_Kinect-Beam
        # replace('ę', 'e') for e. g. 2015-01-28-12-35-21_Kinect-Beam:
        # "es ist beschämend dass dieser umstand herrn pęks aufmerksamkeit entgangen ist"
        # replace('ō', 'o') for e. g. 2015-02-03-13-43-46_Kinect-Beam:
        # "alle varianten von sankyo werden im aikidō üblicherweise in eine immobilisation überführt"
        # replace('á', 'a') for e. g. 2015-02-03-13-45-08_Kinect-Beam:
        # "in dieser hinsicht glaube ich dass sich herr szájer herr swoboda ..."
        # replace('í', 'i') for e. g. 2015-02-09-12-35-23_Kinect-Beam:
        # "das andauernde leben in der einsamkeit zermürbt gísli ..."
        # replace('ł', 'l') for e. g. 2015-02-10-13-41-20_Kinect-Beam:
        # "gegenüber dieser bedrohung gelang dem polnischen führer piłsudski ..."
        # replace('à', 'a') for e. g. 2015-02-10-14-29-03_Kinect-Beam:
        # "... à hundert franken"
        # replace('ė', 'e') in 2015-01-27-13-33-01_Kinect-Beam:
        # "...vom preußischen grenzort laugszargen über tauragė ..."
        # replace('ú','u') in 2015-02-04-13-03-47_Kinect-Beam:
        # "... die von renault in setúbal hergestellte produktlinie ..."
        return super()._decode_german(text).replace("co2", "co zwei").replace('ț', 't'). \
            replace('š', 's').replace('č', 'c').replace('ę', 'e').replace('ō', 'o').replace('á', 'a'). \
            replace('í', 'i').replace('ł', 'l').replace('à', 'a').replace('ė', 'e').replace('ú', 'u')

    def _extract_label_from_xml(self, xml_file: Path) -> str:
        try:
            return self._decode_german(
                ElementTree.parse(str(xml_file)).getroot().find('.//cleaned_sentence').text.lower())
        except Exception:
            raise ParsingException("Error parsing annotation {}".format(xml_file))


def german_corpus(base_directory: Path) -> CombinedCorpus:
    return CombinedCorpus(
        clarin_corpora_sorted_by_size(base_directory=base_directory) + \
        [GermanVoxforgeCorpus(base_directory=base_directory)])
