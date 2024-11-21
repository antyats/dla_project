from pathlib import Path

from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import read_json, write_json


class CustomDataset(BaseDataset):
    def __init__(self, data_dir, part, *args, **kwargs):
        if type(data_dir) is str:
            data_dir = Path(data_dir)
        assert data_dir.is_dir(), f"The folder {data_dir} does not exist"

        self.index_path = data_dir / "index.json"

        if self.index_path.exists():
            index = read_json(str(self.index_path))
        else:
            index = self._create_index(data_dir, part)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, data_dir, part):
        index = []
        audio_files = [
            path
            for path in (data_dir / "audio" / "mix").iterdir()
            if path.is_file() and path.suffix in [".wav", ".flac", ".mp3"]
        ]
        for path in tqdm(audio_files, desc="Indexing custom dataset"):
            entry = {}
            entry["mix_audio_path"] = str(path)

            speaker1_id, speaker2_id = path.stem.split("_")
            entry["speaker1_video_path"] = str(
                data_dir / "mouths" / (speaker1_id + ".npz")
            )
            entry["speaker2_video_path"] = str(
                data_dir / "mouths" / (speaker2_id + ".npz")
            )

            entry["speaker1_audio_path"] = data_dir / "audio" / "s1" / path.name
            entry["speaker2_audio_path"] = data_dir / "audio" / "s2" / path.name

            if (
                entry["speaker1_audio_path"].exists()
                and entry["speaker2_audio_path"].exists()
            ):
                entry["speaker1_audio_path"] = str(entry["speaker1_audio_path"])
                entry["speaker2_audio_path"] = str(entry["speaker2_audio_path"])
            else:
                entry["speaker1_audio_path"] = None
                entry["speaker2_audio_path"] = None

            index.append(entry)

        write_json(index, str(self.index_path))
        return index
