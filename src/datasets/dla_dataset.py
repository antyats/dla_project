from pathlib import Path

from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import read_json, write_json

PARTS = ["test", "train", "val"]


class DLADataset(BaseDataset):
    def __init__(self, data_dir, part, *args, **kwargs):
        if type(data_dir) is str:
            data_dir = Path(data_dir)
        assert data_dir.is_dir(), f"The folder {data_dir} does not exist"
        assert part in PARTS, f"No such part as '{part}'"

        self.index_path = data_dir / "audio" / part / "index.json"

        if self.index_path.exists():
            index = read_json(str(self.index_path))
        else:
            index = self._create_index(data_dir, part)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, data_dir, part):
        index = []
        for path in tqdm(
            [
                f
                for f in Path(data_dir / "audio" / part / "mix").iterdir()
                if f.is_file()
            ],
            desc=f"indexing dla dataset. Part: {part}",
        ):
            entry = {}
            if path.suffix in [".wav", ".flac", ".mp3"]:
                entry["mix_audio_path"] = str(path)
                speaker1_id, speaker2_id = path.stem.split("_")
                entry["speaker1_video_path"] = str(
                    data_dir / "mouths" / (speaker1_id + ".npz")
                )
                entry["speaker2_video_path"] = str(
                    data_dir / "mouths" / (speaker2_id + ".npz")
                )

                entry["speaker1_audio_path"] = (
                    data_dir / "audio" / part / "s1" / path.name
                )
                entry["speaker2_audio_path"] = (
                    data_dir / "audio" / part / "s2" / path.name
                )
                if (
                    entry["speaker1_audio_path"].exists()
                    and entry["speaker2_audio_path"].exists()
                ):
                    entry["speaker1_audio_path"] = str(entry["speaker1_audio_path"])
                    entry["speaker2_audio_path"] = str(entry["speaker2_audio_path"])
                else:
                    entry["speaker1_audio_path"] = None
                    entry["speaker2_audio_path"] = None
            if entry:
                index.append(entry)

        write_json(index, str(self.index_path))
        return index
