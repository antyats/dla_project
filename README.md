# Audio-Visual Source Separation Project

В проекте реализованы 4 модели: CTCNet, TDANet, IIANet, ConvTasNet

ссылка на веса модели (TDFNet): https://drive.google.com/file/d/17Qj1DVkEZ1k1Y1MUrJYbcw0dn4hNQ3By/view?usp=sharing

# Настройка

Для начала нужно положить веса моделей:

 - Предобученные видеомодели в папку *./pretrained_video_models*
 - Веса моделей по source separation кладем в папку *./weights*
 - Если вы хотите использовать датасет из домашнего задания, то запустите *download_dla_dataset.py* и положите его в папку ./data/
 - Если вы хотите использовать свой датасет для инференса, положите его в директорию ./data/ с названием папки inference_dataset
 
 - `pip install -r requirements.txt`

##  Обучение модели

Для того, чтобы обучить модель необходимо выполнить скрипт 

    python train.py -cn=<имя конфига>

Доступные конфиги:
 - ctcnet.yaml
 - tdfnet.yaml
 - iiannet.yaml
 - convtasnet.yaml

## Инференс модели

Для того, чтобы выполнить инференс нашей лучшией модели на ваших данных необходимо выполнить скрипт 

    python inference.py -cn=inference.yaml

Тогда будет произведена подстановка датасета из папки ./data/inference_dataset
