from pathlib import Path
from textwrap import wrap

import librosa
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from numpy import ndarray

from labeled_example import LabeledExample, SpectrogramType, SpectrogramFrequencyScale


class LabeledExamplePlotter:
    def __init__(self, example: LabeledExample):
        self.example = example

    def _plot_audio(self, audio: ndarray) -> None:
        plt.title(str(self))
        plt.xlabel("time / samples (sample rate {}Hz)".format(self.example.sample_rate))
        plt.ylabel("y")
        plt.plot(audio)
        plt.show()

    def show_spectrogram(self, type: SpectrogramType = SpectrogramType.power_level):
        self.prepare_spectrogram_plot(type)
        plt.show()

    def save_spectrogram(self, target_directory: Path,
                         type: SpectrogramType = SpectrogramType.power_level,
                         frequency_scale: SpectrogramFrequencyScale = SpectrogramFrequencyScale.linear) -> Path:
        self.prepare_spectrogram_plot(type, frequency_scale)
        path = Path(target_directory, "{}_{}{}_spectrogram.png".format(
            self.example.id,
            "mel_" if frequency_scale == SpectrogramFrequencyScale.mel else "", type.value.replace(" ", "_")))

        plt.savefig(str(path))
        return path

    def plot_raw_audio(self) -> None:
        self._plot_audio(self.example.raw_audio)

    def prepare_spectrogram_plot(self, type: SpectrogramType = SpectrogramType.power_level,
                                 frequency_scale: SpectrogramFrequencyScale = SpectrogramFrequencyScale.linear) -> None:
        spectrogram = self.example.spectrogram(type, frequency_scale=frequency_scale)

        figure, axes = plt.subplots(1, 1)
        use_mel = frequency_scale == SpectrogramFrequencyScale.mel

        plt.title("\n".join(wrap(
            "{0}{1} spectrogram for {2}".format(("mel " if use_mel else ""), type.value, str(self)), width=100)))
        plt.xlabel("time (data every {}ms)".format(round(1000 / self.example.time_step_rate())))
        plt.ylabel("frequency (data evenly distributed on {} scale, {} total)".format(
            frequency_scale.value, self.example.frequency_count_from_spectrogram(spectrogram)))
        mel_frequencies = self.example.mel_frequencies()
        plt.imshow(
            spectrogram, cmap='gist_heat', origin='lower', aspect='auto', extent=
            [0, self.example.duration_in_s(),
             librosa.hz_to_mel(mel_frequencies[0])[0] if use_mel else 0,
             librosa.hz_to_mel(mel_frequencies[-1])[0] if use_mel else self.example.highest_detectable_frequency()])

        plt.colorbar(label="{} ({})".format(
            type.value,
            "in{} dB, not aligned to a particular base level".format(" something similar to" if use_mel else "") if
            type == SpectrogramType.power_level else "only proportional to physical scale"))

        class ScalarFormatterWithUnit(ScalarFormatter):
            def __init__(self, unit: str):
                super().__init__()
                self.unit = unit

            def __call__(self, x, pos=None) -> str:
                return super().__call__(x, pos) + self.unit

        axes.xaxis.set_major_formatter(ScalarFormatterWithUnit("s"))
        axes.yaxis.set_major_formatter(
            FuncFormatter(lambda value, pos: "{}mel = {}Hz".format(int(value), int(
                librosa.mel_to_hz(value)[0]))) if use_mel else ScalarFormatterWithUnit("Hz"))
        figure.set_size_inches(19.20, 10.80)

    def plot_reconstructed_audio_from_spectrogram(self) -> None:
        self._plot_audio(self.example.reconstructed_audio_from_spectrogram())

    def save_reconstructed_audio_from_spectrogram(self, target_directory: Path) -> None:
        librosa.output.write_wav(
            str(Path(target_directory,
                     "{}_window{}_hop{}.wav".format(self.example.id, self.example.fourier_window_length,
                                                    self.example.hop_length))),
            self.example.reconstructed_audio_from_spectrogram(), sr=self.example.sample_rate)

    def save_spectrograms_of_all_types(self, target_directory: Path) -> None:
        for type in SpectrogramType:
            for frequency_scale in SpectrogramFrequencyScale:
                self.save_spectrogram(target_directory=target_directory, type=type,
                                      frequency_scale=frequency_scale)
