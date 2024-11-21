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
    ).repeat(2, 1, 1)

    batch["mix_audio_len"] = torch.tensor(
        [item["mix_audio_len"] for item in dataset_items]
    ).repeat(2, 1, 1)

    batch["video"] = pad_sequence(
        [item["video1"] for item in dataset_items]
        + [item["video2"] for item in dataset_items],
        batch_first=True,
    ).unsqueeze(1)

    batch["video_len"] = torch.tensor(
        [item["video1_len"] for item in dataset_items]
        + [item["video2_len"] for item in dataset_items]
    )

    if "mix_spectrogram" in dataset_items[0]:
        batch["mix_spectrogram"] = (
            pad_sequence(
                [
                    item["mix_spectrogram"].squeeze(0).transpose(0, 1)
                    for item in dataset_items
                ],
                batch_first=True,
            )
            .transpose(1, 2)
            .repeat(2, 1, 1)
        )

        batch["mix_spectrogram_len"] = torch.tensor(
            [item["mix_spectrogram_len"] for item in dataset_items]
        ).repeat(2, 1, 1)

    if ("speaker1_audio" in dataset_items[0]) and (
        "speaker2_audio" in dataset_items[0]
    ):
        batch["target_audio"] = pad_sequence(
            [item["speaker1_audio"] for item in dataset_items]
            + [item["speaker2_audio"] for item in dataset_items],
            batch_first=True,
        )
        batch["target_audio_len"] = torch.tensor(
            [item["speaker1_audio_len"] for item in dataset_items]
            + [item["speaker2_audio_len"] for item in dataset_items]
        )

    if ("speaker1_spectrogram" in dataset_items[0]) and (
        "speaker2_spectrogram" in dataset_items[0]
    ):
        batch["target_spectrogram"] = pad_sequence(
            [
                item["speaker1_spectrogram"].squeeze(0).transpose(0, 1)
                for item in dataset_items
            ]
            + [
                item["speaker2_spectrogram"].squeeze(0).transpose(0, 1)
                for item in dataset_items
            ],
            batch_first=True,
        ).transpose(1, 2)

        batch["target_spectrogram_len"] = torch.tensor(
            [item["speaker1_spectrogram_len"] for item in dataset_items]
            + [item["speaker2_spectrogram_len"] for item in dataset_items]
        )

    return batch
