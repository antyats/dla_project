import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
        self,
        index,
        target_sr=16000,
        limit=None,
        shuffle_index=False,
        instance_transforms=None,
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_index_is_valid(index)

        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: List[dict] = index

        self.target_sr = target_sr

        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]

        mix_audio_path = data_dict["mix_audio_path"]
        speaker1_audio_path = data_dict["speaker1_audio_path"]
        speaker2_audio_path = data_dict["speaker2_audio_path"]
        speaker1_video_path = data_dict["speaker1_video_path"]
        speaker2_video_path = data_dict["speaker2_video_path"]

        mix_audio = self.load_audio(mix_audio_path)
        speaker1_video = self.load_video(speaker1_video_path)
        speaker2_video = self.load_video(speaker2_video_path)

        mix_audio = self.wave_augmentation(mix_audio)
        mix_spectrogram = self.get_spectrogram(mix_audio)
        mix_spectrogram = self.spectrogram_augmentation(mix_spectrogram)

        speaker1_video = self.video_augmentation(speaker1_video)
        speaker2_video = self.video_augmentation(speaker2_video)

        instance_data = {
            "mix_audio": mix_audio,
            "mix_audio_len": mix_audio.size(-1),
            "mix_spectrogram": mix_spectrogram,
            "mix_spectrogram_len": (
                mix_spectrogram.size(-1) if mix_spectrogram is not None else None
            ),
            "video1": speaker1_video,
            "video1_len": speaker1_video.size(0),
            "video2": speaker2_video,
            "video2_len": speaker2_video.size(0),
        }
        if (speaker1_audio_path and speaker1_audio_path) is not None:
            speaker1_audio = self.load_audio(speaker1_audio_path)
            speaker2_audio = self.load_audio(speaker2_audio_path)
            speaker1_spectrogram = self.get_spectrogram(speaker1_audio)
            speaker2_spectrogram = self.get_spectrogram(speaker2_audio)

            instance_data.update(
                {
                    "speaker1_audio": speaker1_audio,
                    "speaker1_audio_len": speaker1_audio.size(-1),
                    "speaker1_spectrogram": speaker1_spectrogram,
                    "speaker1_spectrogram_len": (
                        speaker1_spectrogram.size(-1)
                        if speaker1_spectrogram is not None
                        else None
                    ),
                    "speaker2_audio": speaker2_audio,
                    "speaker2_audio_len": speaker2_audio.size(-1),
                    "speaker2_spectrogram": speaker2_spectrogram,
                    "speaker2_spectrogram_len": (
                        speaker2_spectrogram.size(-1)
                        if speaker2_spectrogram is not None
                        else None
                    ),
                }
            )

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

    def load_audio(self, path):
        """
        Load audio from disk.

        Args:
            path(str): path to the audio (wav/flac/mp3).
        Returns:
            Audio tensor.
        """
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def load_video(self, path):
        video = np.load(path)
        video_tensor = torch.from_numpy(video["data"])
        return video_tensor / 255

    def get_spectrogram(self, audio):
        if self.instance_transforms is not None:
            if "get_spectrogram" in self.instance_transforms:
                return self.instance_transforms["get_spectrogram"](audio)
        return None

    def wave_augmentation(self, audio):
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                if transform_name == "audio":
                    audio = self.instance_transforms[transform_name](audio)
        return audio

    def spectrogram_augmentation(self, spectrogram):
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                if transform_name == "spectrogram":
                    spectrogram = self.instance_transforms[transform_name](spectrogram)
        return spectrogram

    def video_augmentation(self, video):
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                if transform_name == "video":
                    video = self.instance_transforms[transform_name](video)
        return video

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert (
                "mix_audio_path" in entry
            ), "Each dataset item should include field 'mix_audio_path' - path to audio file."
            assert (
                "speaker1_video_path" in entry
            ), "Each dataset item should include field 'speaker1_video_path' - path to file, that contains video information for the speaker1."
            assert (
                "speaker2_video_path" in entry
            ), "Each dataset item should include field 'speaker2_video_path' - path to file, that contains video information for the speaker2."
            assert (
                "speaker1_audio_path" in entry
            ), "Each dataset item should include field 'speaker1_audio_path' - None / path to file, that contains ground truth for the speaker1"
            assert (
                "speaker2_audio_path" in entry
            ), "Each dataset item should include field 'speaker2_audio_path' - None / path to file, that contains ground truth for the speaker2"

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
