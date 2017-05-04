import array
from itertools import dropwhile
from pathlib import Path
from sys import byteorder

import librosa
import numpy
from numpy import ndarray, abs, max, flipud, concatenate

from speechless import configuration
from speechless.labeled_example import LabeledExample
from speechless.tools import timestamp, mkdir


class Recorder:
    def __init__(self,
                 silence_threshold_for_unnormalized_audio: float = .03,
                 chunk_size: int = 1024,
                 sample_rate: int = 16000,
                 silence_until_terminate_in_s: int = 3):
        self.silence_threshold_for_not_normalized_sound = silence_threshold_for_unnormalized_audio
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self.silence_until_terminate_in_s = silence_until_terminate_in_s

    def _is_silent(self, audio: ndarray):
        return max(audio) < self.silence_threshold_for_not_normalized_sound

    def _normalize(self, audio: ndarray) -> ndarray:
        return audio / max(abs(audio))

    def _trim_silence(self, audio: ndarray) -> ndarray:
        def trim_start(sound: ndarray) -> ndarray:
            return numpy.array(list(dropwhile(lambda x: x < self.silence_threshold_for_not_normalized_sound, sound)))

        def trim_end(sound: ndarray) -> ndarray:
            return flipud(trim_start(flipud(sound)))

        return trim_start(trim_end(audio))

    def record(self):
        """Records from the microphone and returns the data as an array of signed shorts."""

        print("Wait in silence to begin recording; wait in silence to terminate")

        import pyaudio

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate, input=True, output=True,
                        frames_per_buffer=self.chunk_size)

        silent_chunk_count = 0
        has_recording_started = False
        is_first_chunk = False
        chunks = []

        while True:
            chunk_as_array = array.array('f', stream.read(self.chunk_size))

            # drop first, as it is often loud noise
            if not is_first_chunk:
                is_first_chunk = True
                continue

            if byteorder == 'big':
                chunk_as_array.byteswap()

            chunk = numpy.array(chunk_as_array)

            chunks.append(chunk)

            silent = self._is_silent(chunk)
            print("Silent: " + str(silent))

            if has_recording_started:
                if silent:
                    silent_chunk_count += 1
                    if silent_chunk_count * self.chunk_size > self.silence_until_terminate_in_s * self.sample_rate:
                        break
                else:
                    silent_chunk_count = 0
            elif not silent:
                has_recording_started = True

        stream.stop_stream()
        stream.close()
        print("Stopped recording.")

        p.terminate()

        return self._normalize(self._trim_silence(concatenate(chunks)))

    def record_to_file(self, path: Path) -> LabeledExample:
        "Records from the microphone and outputs the resulting data to 'path'. Returns a labeled example for analysis."
        librosa.output.write_wav(str(path), self.record(), self.sample_rate)

        return LabeledExample(path)


def record_plot_and_save(
        recording_directory: Path = configuration.default_data_directories.recording_directory) -> LabeledExample:
    from speechless.labeled_example_plotter import LabeledExamplePlotter

    mkdir(recording_directory)
    name = "recording-{}".format(timestamp())
    example = Recorder().record_to_file(recording_directory / "{}.wav".format(name))
    LabeledExamplePlotter(example).save_spectrogram(recording_directory)

    return example
