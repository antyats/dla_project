# DLA project: Audio-Visual Source Separation

## About
This repository provides implementations of several models for audio-visual source separation. The models in this repository are heavily based on the following papers (however, they are not strict implementations of them):



*   [CTCNet](https://arxiv.org/pdf/2212.10744)
*   [TDFNet](https://arxiv.org/pdf/2401.14185)



## Instalation
1. Clone the repository:


```bash
git clone https://github.com/antyats/dla_project.git
cd dla_project
```

2. Download lip-reading model [weights](https://bit.ly/3AQTFOG) (we used resnet18_mstcn_video from [this](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks) repository). You can do it manually or by running following commands:

```bash
pip install gdown
mkdir pretrained_video_models
gdown https://drive.google.com/uc?id=1vqMpxZ5LzJjg50HlZdj_QFJGm2gQmDUD --fuzzy --output
"./pretrained_video_models/"
```

3. Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n dla_project_env python=PYTHON_VERSION

   # activate env
   conda activate dla_project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv dla_project_env

   # alternatively, using default python version
   python3 -m venv dla_project_env

   # activate env
   source dla_project_env
   ```

4. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

## Data

If you want to use DLA dataset, you can download it manually or by running the following command:

```bash
python3 download_dla_dataset.py [-h] [--extract] url [dir]
```

Or you can use any custom dataset with the following structure (using class src.datasets.CustomDataset):


```bash
NameOfTheDirectoryWithUtterances
├── audio
│   ├── mix
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   ├── s1 # ground truth for the speaker s1, may not be given
│   │   ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│   │   ├── FirstSpeakerID2_SecondSpeakerID2.wav
│   │   .
│   │   .
│   │   .
│   │   └── FirstSpeakerIDn_SecondSpeakerIDn.wav
│   └── s2 # ground truth for the speaker s2, may not be given
│       ├── FirstSpeakerID1_SecondSpeakerID1.wav # also may be flac or mp3
│       ├── FirstSpeakerID2_SecondSpeakerID2.wav
│       .
│       .
│       .
│       └── FirstSpeakerIDn_SecondSpeakerIDn.wav
└── mouths # contains video information for all speakers
    ├── FirstOrSecondSpeakerID1.npz # npz mouth-crop
    ├── FirstOrSecondSpeakerID2.npz
    .
    .
    .
    └── FirstOrSecondSpeakerIDn.npz
```


## Models

Our best models perfomance

| Model | Dataset | SI-SNRi | SDRi | PESQ | STOI | config | weights | wandb run |
|:-|:-|:-|:-|:-|:-|:-|:-|:-|
| TDFNet | DLA | 10.72 | 11.14 | 1.94 | 0.88 | [.yml](src/configs/model/tdfnet.yaml)| [link](https://drive.google.com/file/d/17Qj1DVkEZ1k1Y1MUrJYbcw0dn4hNQ3By/view)|[part1](https://wandb.ai/max23-ost/dla_avss_project_test/runs/pv0duxhx?nw=nwusermax23ost), [part2](https://wandb.ai/max23-ost/dla_avss_project_test/runs/9iss8i49?nw=nwusermax23ost), [part3](https://wandb.ai/max23-ost/dla_avss_project_test/runs/hnzeh0b7?nw=nwusermax23ost)|
| CTCNet | DLA | 10.42 | 7.72 | 1.90 | 0.87 | [.yml](src/configs/model/ctcnet.yaml)| [link](https://drive.google.com/file/d/1_qB92RWSHj6K0ljUoYNO50BynTcdVS0Y/view)|[run](https://wandb.ai/max23-ost/dla_avss_project_test/runs/sniolv3v?nw=nwusermax23ost)|

*There were 3 runs for TDFNet as one training procedure.*


## How To Use

###To run inference###

```bash
python3 inference.py \
datasets.test.data_dir=<PATH/TO/CUSTOM/DATASET/DIR> \
inferencer.save_path=<PATH/TO/SAVE/PREDICTIONS/TO> \
inferencer.from_pretrained=<PATH/TO/PREATRAINED/MODEL> \
model.path_to_pretrained_video_extractor=<PATH/TO/PRETRAINED/LIPREADING/MODEL>
```
Add this at the end of the command if you want to evaluate the metrics
```bash
+metrics=avss_metrics
```
###To calculate metrics###
Use `compute_metrics.py` script. You need to pass 3 arguments:
1. \<preds\-path\> - path to predictions folder. It must contain `s1` and `s2` folders with `wav`-files in it.
2. \<ground-truth-path\> - path to folder with ground truth audios. It must contain `s1` and `s2` folders with `wav`-files in it.
3. \<source-path\> - path to folder with source audios. It must contain `mix` folder with `wav`-files in it. (source audios are needed for `SI-SNRi` and `SDRi` metrics computations)

```bash
python3 compute_metrics.py <preds-path> <ground-truth-path> <source-path>
```
