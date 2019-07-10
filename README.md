# Instrutions

## Setup

- If use GPU, `export GPU_ENABLE=1`
- script/docker/setup.sh
- Install VNC viewer

## Start

- script/docker/start.sh
- docker exec -it demoimageclassification_main bash
- Open VNC viewer with address 0.0.0.0:5900 to see camera

## Training

- Run `python train_model.py`
  - options:
    - --dataset: path to input dataset, default: dataset
    - --model: training model (letnet or minivggnet), default: minivggnet
    - --output: path to output model, default: output/minivggnet.h5
    - --reset: value: 1 - capture images then train, value: 0 - train with current dataset
- Capture pictures:
  - Press SPACE to capture pictures for current class (in camera window)
  - Press SHIFT to move to next class (in camera window)
  - Press ENTER to start training (in camera window)
  - Press Esc to quit (in camera window)
- Tips:
  - Take at least 100 images per class
  - With images number per class less than 1000, I prefer model lenet
  - With images number per class less than 1000, I prefer model minivggnet

## Tesing

- Run `python test_model.py`
  - options:
    - --dataset: path to input dataset, default: dataset
    - --model: training model (letnet or minivggnet), default: minivggnet
- Point camara to object
- Dected Image window will show object with highest match score
- Press Esc to quit (in camera window)
