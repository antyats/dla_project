import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    batch = {}
    batch["mix_audio"] = pad_sequence(
        [item["mix_audio"].squeeze(0) for item in dataset_items], batch_first=True
    )
    batch["mix_audio_len"] = torch.tensor(
        [item["mix_audio_len"] for item in dataset_items]
    )

    batch["video1"] = pad_sequence(
        [item["video1"] for item in dataset_items], batch_first=True
    )
    batch["video1_len"] = torch.tensor([item["video1_len"] for item in dataset_items])

    batch["video2"] = pad_sequence(
        [item["video2"] for item in dataset_items], batch_first=True
    )
    batch["video2_len"] = torch.tensor([item["video2_len"] for item in dataset_items])

    if "mix_spectrogram" in dataset_items[0]:
        batch["mix_spectrogram"] = pad_sequence(
            [
                item["mix_spectrogram"].squeeze(0).transpose(0, 1)
                for item in dataset_items
            ],
            batch_first=True,
        ).transpose(1, 2)
        batch["mix_spectrogram_len"] = torch.tensor(
            [item["mix_spectrogram_len"] for item in dataset_items]
        )

    if ("speaker1_audio" in dataset_items[0]) and (
        "speaker2_audio" in dataset_items[0]
    ):
        batch["speaker1_audio"] = pad_sequence(
            [item["speaker1_audio"].squeeze(0) for item in dataset_items],
            batch_first=True,
        )
        batch["speaker1_audio_len"] = torch.tensor(
            [item["speaker1_audio_len"] for item in dataset_items]
        )

        batch["speaker2_audio"] = pad_sequence(
            [item["speaker2_audio"].squeeze(0) for item in dataset_items],
            batch_first=True,
        )
        batch["speaker2_audio_len"] = torch.tensor(
            [item["speaker2_audio_len"] for item in dataset_items]
        )

    if ("speaker1_spectrogram" in dataset_items[0]) and (
        "speaker2_spectrogram" in dataset_items[0]
    ):
        batch["speaker1_spectrogram"] = pad_sequence(
            [
                item["speaker1_spectrogram"].squeeze(0).transpose(0, 1)
                for item in dataset_items
            ],
            batch_first=True,
        ).transpose(1, 2)
        batch["speaker1_spectrogram_len"] = torch.tensor(
            [item["speaker1_spectrogram_len"] for item in dataset_items]
        )

        batch["speaker2_spectrogram"] = pad_sequence(
            [
                item["speaker2_spectrogram"].squeeze(0).transpose(0, 1)
                for item in dataset_items
            ],
            batch_first=True,
        ).transpose(1, 2)
        batch["speaker2_spectrogram_len"] = torch.tensor(
            [item["speaker2_spectrogram_len"] for item in dataset_items]
        )

    return batch
