import argparse
import torchaudio
from tqdm import tqdm
from pathlib import Path

from src.metrics import PESQ, SDRi, SI_SNRi, STOI

YANDEX_API_URL = "https://cloud-api.yandex.net/v1/disk/public/resources/download"

pesq = PESQ(sample_rate=16000)
sdri = SDRi()
si_snri = SI_SNRi()
stoi = STOI(sample_rate=16000)


def get_metrics(mix_audio, pred_audio, gt_audio):
    return {
        "PESQ": pesq(pred_audio, gt_audio),
        "SDRi": sdri(mix_audio, pred_audio, gt_audio),
        "SI_SNRi": si_snri(mix_audio, pred_audio, gt_audio),
        "STOI": stoi(pred_audio, gt_audio),
    }


def main(args):
    preds_folder_contents = sorted((args.preds / "s1").iterdir())
    l = len(preds_folder_contents)
    total_metrics = {"PESQ": 0, "SDRi": 0, "SI_SNRi": 0, "STOI": 0}

    for wav in tqdm(preds_folder_contents, desc="calculating metrics"):
        for speaker in ["s1", "s2"]:
            pred_wav = args.preds / speaker / wav.name
            gt_wav = args.ground_truth / speaker / wav.name
            source_wav = args.source / "mix" / wav.name

            assert wav.is_file()
            assert gt_wav.is_file()
            assert source_wav.is_file()

            pred_audio, _ = torchaudio.load(pred_wav)
            gt_audio, _ = torchaudio.load(gt_wav)
            mix_audio, _ = torchaudio.load(source_wav)

            metrics = get_metrics(mix_audio, pred_audio, gt_audio)
            for key in total_metrics.keys():
                total_metrics[key] += metrics[key]

    for key in total_metrics.keys():
        total_metrics[key] = total_metrics[key] / (l * 2)  # as 2 speakers
        print(f"{key}: {total_metrics[key]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute metrics for predicted wavs")
    parser.add_argument(
        "preds", type=Path, help="The folder with predictions (`s1` and `s2` folders)"
    )
    parser.add_argument(
        "ground_truth",
        type=Path,
        help="The folder with ground truth (`s1` and `s2` folders)",
    )
    parser.add_argument(
        "source",
        type=Path,
        help="The folder with source audios (with `mix` folder). Needed for SDRi and SI-SNRi",
    )

    args = parser.parse_args()
    main(args)
