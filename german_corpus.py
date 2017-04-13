import json
import re
from pathlib import Path

from collections import OrderedDict
from typing import Iterable, Dict, Callable, Optional, List, Tuple, Set
from xml.etree import ElementTree

from corpus import ParsingException, TrainingTestSplit, CombinedCorpus
from english_corpus import LibriSpeechCorpus
from grapheme_enconding import frequent_characters_in_german
from labeled_example import LabeledExample, PositionalLabel
from tools import read_text, single, single_or_none, name_without_extension, group

_tags_to_ignore = [
    "<usb>",  # truncated in beginning or incomprehensible
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
                     List[LabeledExample], List[LabeledExample]]] = TrainingTestSplit.randomly_grouped_by_directory()):
        self.umlaut_decoder = umlaut_decoder

        print("Parsing corpus {}...".format(corpus_name))

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
                         training_test_split=training_test_split,
                         maximum_example_duration_in_s=35)

    def _extract_positional_label_by_id(self, files: Iterable[Path]) -> Dict[str, PositionalLabel]:
        json_ending = "_annot.json"
        json_annotation_files = \
            [file for file in files if file.name.endswith(json_ending) and
             self.id_filter_regex.match(file.name[:-len(json_ending)])]

        json_extracted = OrderedDict(
            (file.name[:-len(json_ending)], self._extract_positional_label_from_json(file)) for
            file
            in json_annotation_files)

        par_annotation_files = [file for file in files if file.name.lower().endswith(".par")
                                and self.id_filter_regex.match(name_without_extension(file).lower())]

        extracted = OrderedDict(
            (name_without_extension(file), self._extract_positional_label_from_par(file)) for file in
            par_annotation_files)

        for key in set(extracted.keys()).intersection(set(json_extracted.keys())):
            if extracted[key].words != json_extracted[key].words:
                print('{}: Words {} extracted from par differ from json {}'.format(key, extracted[key].words,
                                                                                   json_extracted[key].words))

        # json has positional information and overrides par
        extracted.update(json_extracted)

        # TODO refactor
        if len(self.corpus_names) == 1 and ("ALC" in single(self.corpus_names)):
            # exactly half have no label: can be fixed by using 0061006007_h_00.par or _annot.json instead of 0061006007_m_00_annot.json etc.
            correctly_labeled_id_marker = "_h_"
            empty_labeled_id_marker = "_m_"

            correct_ids = [id for id in extracted.keys() if correctly_labeled_id_marker in id]
            for correct_id in correct_ids:
                empty_labeled_id = correct_id.replace(correctly_labeled_id_marker, empty_labeled_id_marker)
                extracted[empty_labeled_id] = extracted[correct_id]

        return extracted

    def _extract_positional_label_from_json(self, json_file: Path) -> PositionalLabel:
        json_text = read_text(json_file, encoding='utf8')

        try:
            j = json.loads(json_text)
            levels = j["levels"]

            def words_with_id_for_labels(label_names: Set[str]) -> List[Tuple[str, int]]:
                def is_level_empty(level: json) -> bool:
                    return len(level["items"]) == 0

                def is_level_useful(level: json) -> bool:
                    if is_level_empty(level):
                        return False

                    return any([label for label in level["items"][0]["labels"] if label["name"] in label_names])

                def word_with_id(transcription: json) -> Tuple[str, int]:
                    labels = transcription["labels"]

                    matching_labels = [label for label in labels if label["name"] in label_names]

                    if len(matching_labels) == 0:
                        raise Exception("No matching label names, found {} instead.".format(
                            [label["name"] for label in labels]))

                    matching_label = single(matching_labels)
                    return matching_label["value"], transcription["id"]

                words_with_id = single_or_none([[word_with_id(transcription) for
                                                 transcription in level["items"]] for level in levels if
                                                is_level_useful(level)])

                if words_with_id is None:
                    return []

                return words_with_id

            words_with_id = words_with_id_for_labels(label_names={"ORT", "word"})
            tr2_words_with_id = words_with_id_for_labels(label_names={"TR2"})

            ids = [id for word, id in words_with_id]
            words = self._merge_transcriptions_and_decode(words=[word for word, id in words_with_id],
                                                          tr2_words=[word for word, id in tr2_words_with_id])

            segment_ids_by_word_id = group(j["links"], key=lambda link: link["fromID"], value=lambda link: link["toID"])

            def sampel_range_by_segment_id(level_names: Iterable[str]) -> Dict[int, Tuple[int, int]]:
                return OrderedDict(
                    (segment["id"], (segment["sampleStart"], segment["sampleStart"] + segment["sampleDur"] + 1))
                    for level in levels
                    if level["type"] == "SEGMENT" and level["name"] in level_names
                    for segment in level["items"])

            mas_sample_range_by_segment_id = sampel_range_by_segment_id(level_names=("MAS",))
            mau_sample_range_by_segment_id = sampel_range_by_segment_id(level_names=("MAU",))
            pho_sample_range_by_segment_id = sampel_range_by_segment_id(level_names=("PHO", "phonetic"))

            def sampel_ranges_by_word_id(id: int) -> List[Tuple[int, int]]:
                segment_ids = segment_ids_by_word_id[id] if id in segment_ids_by_word_id else []

                def a(x):
                    return [x[segment_id]
                            for segment_id in segment_ids
                            if segment_id in x]

                mas_sample_ranges = a(mas_sample_range_by_segment_id)
                mau_sample_ranges = a(mau_sample_range_by_segment_id)
                pho_sample_ranges = a(pho_sample_range_by_segment_id)

                return pho_sample_ranges if pho_sample_ranges else (
                    mas_sample_ranges if mas_sample_ranges else mau_sample_ranges)

            def merge_consecutive_ranges(ranges: List[Tuple[int, int]]) -> Tuple[int, int]:
                def is_not_empty(range: Tuple[int, int]):
                    return range[0] + 1 != range[1]

                s = sorted((range for range in ranges if is_not_empty(range)), key=lambda range: range[0])[:-1]
                for index, range in enumerate(s):
                    next_range = ranges[index + 1]

                    if range[1] != next_range[0]:
                        print("Ranges {} of a word are not consecutive.".format(s))

                return ranges[0][0], ranges[-1][1]

            def sample_range_or_none_by_word_id(id: int):
                ranges = sampel_ranges_by_word_id(id)

                return merge_consecutive_ranges(ranges) if ranges else None

            return PositionalLabel([(word, sample_range_or_none_by_word_id(id)) for word, id in zip(words, ids)])

        except Exception:
            raise ParsingException("Error parsing annotation {}: {}".format(json_file, json_text[:500]))

    def _extract_positional_label_from_par(self, par_file: Path) -> PositionalLabel:
        par_text = ""

        try:
            par_text = read_text(par_file, encoding="utf8")

            def words_for_label(label_name: str):
                return [line.split("\t")[-1] for line in par_text.splitlines() if line.startswith(label_name)]

            return PositionalLabel(
                [(word, None) for word in
                 self._merge_transcriptions_and_decode(words_for_label("ORT"), words_for_label("TR2"))])
        except Exception:
            raise ParsingException("Error parsing annotation {}: {}".format(par_file, par_text[:500]))

    def _merge_transcriptions_and_decode(self, words: List[str], tr2_words: List[str]) -> List[str]:
        usb_tag = "<usb>"

        def clean_tr2(tr2_word):
            return tr2_word.replace('<Ger"ausch>', '').replace('<geräusch>', '').replace('<#>', '')

        # In the ZIPTEL corpus, ORT or word transcription often contains <usb> tags,
        # while TR2 contains the truncated words instead, e. g. somethi~
        # for better character recognition we use the latter.
        # Why is TR2 not always used? Because it contains extra whitespace and more tags.
        if len(words) > 0:
            if words[0] == usb_tag:
                words[0] = clean_tr2(tr2_words[0])

            if words[-1] == usb_tag:
                if len(tr2_words) != len(words):
                    raise ParsingException("TR2 word count differs.")
                words[-1] = clean_tr2(tr2_words[-1])

        return [self._correct_german(word) for word in words]

    def _correct_german(self, text: str) -> str:
        # replace('é', 'e') because of TODO
        # replace('xe4', 'ä') because of F09S1MP-Mikro_Prompt_20 (+7 more): " timo hat b  xe4ten gesagt"
        # replace('.', ' ') because of ALC: 5204018034_h_00 contains "in l.a."
        # replace('-', ' ') because of some examples in e. g. ZIPTEL, PD2, SC10 like the following:
        # SC10: awed5070: "darf ich eine ic-fahrt zwischendurch unterbrechen"
        return self.umlaut_decoder(
            text.lower().replace('é', 'e').replace('xe4', 'ä').replace('.', ' ').replace('-', ' '))


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

# example fiw1e020 from SC10 corpus has a wrong label (.par/.json is also inconsistent), exclude it:
sc10_broken_label_filter_regex = re.compile("(?!^fiw1e020$)[\s\S]*")


def clarin_corpora_sorted_by_size(base_directory: Path) -> List[GermanClarinCorpus]:
    return [
        sc1(base_directory),
        pd2(base_directory),
        ziptel(base_directory),
        sc10(base_directory),
        GermanClarinCorpus("all.HEMPEL.4.cmdi.11610.1490680796", base_directory),
        GermanClarinCorpus("all.PD1.3.cmdi.16312.1490681066", base_directory),
        GermanClarinCorpus("all.VM1.3.cmdi.1508.1490625070", base_directory,
                           id_filter_regex=vm1_id_german_filter_regex,
                           training_test_split=TrainingTestSplit.training_only),
        GermanClarinCorpus("all.RVG-J.1.cmdi.18181.1490681704", base_directory),
        GermanClarinCorpus("all.ALC.4.cmdi.16602.1490632862", base_directory,
                           training_test_split=TrainingTestSplit.randomly_grouped_by(lambda e: e.id[:3])),
        GermanClarinCorpus("all.VM2.3.cmdi.4260.1490625316", base_directory,
                           id_filter_regex=vm2_id_german_filter_regex,
                           training_test_split=TrainingTestSplit.training_only)
    ]


def sc1(base_directory: Path) -> GermanClarinCorpus:
    return GermanClarinCorpus("all.SC1.3.cmdi.15010.1490631864", base_directory,
                              umlaut_decoder=UmlautDecoder.quote_after_umlaut,
                              training_test_split=TrainingTestSplit.test_only)


def pd2(base_directory: Path) -> GermanClarinCorpus:
    return GermanClarinCorpus("all.PD2.4.cmdi.16693.1490681127", base_directory)


def ziptel(base_directory: Path) -> GermanClarinCorpus:
    return GermanClarinCorpus("all.ZIPTEL.3.cmdi.63058.1490624016", base_directory)


def sc10(base_directory: Path) -> GermanClarinCorpus:
    return GermanClarinCorpus("all.SC10.4.cmdi.13781.1490631055", base_directory,
                              umlaut_decoder=UmlautDecoder.try_quote_before_umlaut_then_after,
                              training_test_split=TrainingTestSplit.test_only,
                              id_filter_regex=sc10_broken_label_filter_regex)


class GermanVoxforgeCorpus(GermanClarinCorpus):
    def __init__(self, base_directory: Path):
        super().__init__(
            corpus_name="german-speechdata-package-v2",
            base_directory=base_directory,
            base_source_url_or_directory="http://www.repository.voxforge1.org/downloads/de/",
            tar_gz_extension=".tar.gz",
            subdirectory_depth=1,
            umlaut_decoder=UmlautDecoder.none,
            training_test_split=TrainingTestSplit.by_directory(),
            tags_to_ignore=[],
            # exclude those 7 audio files because the first 2 are corrupt, the last 5 are empty:
            id_filter_regex=re.compile("(?!^2014-03-24-13-39-24_Kinect-RAW)"
                                       "(?!^2014-03-27-11-50-33_Kinect-RAW)"
                                       "(?!^2014-03-18-15-34-19_Realtek)"
                                       "(?!^2014-06-17-13-46-27_Kinect-RAW)"
                                       "(?!^2014-06-17-13-46-27_Realtek)"
                                       "(?!^2014-06-17-13-46-27_Samson)"
                                       "(?!^2014-06-17-13-46-27_Yamaha)"
                                       "(^.*$)"))

    def _extract_positional_label_by_id(self, files: Iterable[Path]) -> Dict[str, PositionalLabel]:
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

        return OrderedDict(
            (name_without_extension(file) + microphone_ending,
             PositionalLabel.without_positions(self._extract_label_from_xml(file)))
            for file in xml_files
            for microphone_ending in microphone_endings
            if (Path(file.parent) / (name_without_extension(file) + microphone_ending + ".wav")).exists())

    def _correct_german(self, text: str) -> str:
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
        replaced = super()._correct_german(text).replace("co2", "co zwei").replace('ț', 't').replace('š', 's').replace(
            'č', 'c').replace('ę', 'e').replace('ō', 'o').replace('á', 'a').replace('í', 'i').replace('ł', 'l').replace(
            'à', 'a').replace('ė', 'e').replace('ú', 'u')

        return replaced

    def _extract_label_from_xml(self, xml_file: Path) -> str:
        try:
            return self._correct_german(
                ElementTree.parse(str(xml_file)).getroot().find('.//cleaned_sentence').text.lower())
        except Exception:
            raise ParsingException("Error parsing annotation {}".format(xml_file))


def german_corpus(base_directory: Path) -> CombinedCorpus:
    return CombinedCorpus(
        clarin_corpora_sorted_by_size(base_directory=base_directory) +
        [GermanVoxforgeCorpus(base_directory=base_directory)])
