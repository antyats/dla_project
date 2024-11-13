from src.datasets.base_dataset import BaseDataset

class DLADataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        assert data_dir.is_dir(), f"The folder {data_dir} does not exist"

        index = []
        for path in (data_dir / "audio" / "mix").iterdir():
            entry = {}
            if path.suffix in [".waw", ".flac", ".mp3"]:
                entry["mix_audio_path"] = str(path)
                speaker1_id, speaker2_id = path.stem.split('_')
                entry["speaker1_video_path"] = str(data_dir / "video" / (speaker1_id + ".npz"))
                entry["speaker2_video_path"] = str(data_dir / "video" / (speaker2_id + ".npz"))

                entry["speaker1_audio_path"] = data_dir / "audio" / "s1" / path.name
                entry["speaker2_audio_path"] = data_dir / "audio" / "s2" / path.name
                if entry["speaker1_audio_path"].exists() and entry["speaker2_audio_path"].exists():
                    entry["speaker1_audio_path"] = str(entry["speaker1_audio_path"])
                    entry["speaker2_audio_path"] = str(entry["speaker2_audio_path"])
                else:
                    entry["speaker1_audio_path"] = None
                    entry["speaker2_audio_path"] = None
            if entry:
                index.append(entry)
        super().__init__(index, *args, **kwargs)
