# Instrutions

## Setup

- Run 'script/docker/setup.sh'
- Run 'script/docker/run.sh bash'

## Training

- Run `python3.7 train_model.py`
  - options:
    - --dataset: path to input dataset, default: dataset
    - --model: training model (letnet or minivggnet), default: lenet
    - --output: path to output model, default: output/lenet.hdf5
    - --reset: value: 1 - capture images then train, value: 0 - train with current dataset, default: 1
- Capture pictures:
  - Press SPACE to capture pictures for current class
  - Press SHIFT to move to next class
  - Press ENTER to start training
  - Press Esc to quit
- Tips:
  - Take at least 100 images per class
  - With images number per class less than 1000, I prefer model lenet
  - With images number per class less than 1000, I prefer model minivggnet

## Tesing

- Run `python3.7 test_model.py`
- Point camara to object
- Dected Image window will show object with highest match score
