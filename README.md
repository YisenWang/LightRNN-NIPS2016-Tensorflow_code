# LightRNN-NIPS2016-Tensorflow_code
The tensorflow implementation of NIPS2016 paper "LightRNN: Memory and Computation-Efficient Recurrent Neural Networks" (https://arxiv.org/abs/1610.09893)

## Dependencies
- Python 2.7
- Tensorflow 0.12
- ortools 5.0.3919 (https://github.com/google/or-tools)

## Usage
python train_lm.py --data_path=./data/

## Others
- For minimum weight perfect matching algorithm, we use the minimum cost flow solver in ortools. 
- Currently, we test the code on PTB dataset.
- This is a initital version. If you find any problems, please don't hesitate to contact me: eewangyisen AT gmail DOT com
