# LightRNN-NIPS2016-Tensorflow_code
The tensorflow implementation of NIPS2016 paper "LightRNN: Memory and Computation-Efficient Recurrent Neural Networks" (https://arxiv.org/abs/1610.09893)

## LightRNN: Memory and Computation-Efficient Recurrent Neural Networks
To address the both model size and running time, especially for text corpora with large vocabularies, the author proposed to use 2-Component (2C) shared embedding for word representations. They allocate every word in the vocabulary into a table, each row of which is associated with a vector, and each column associated with another vector. Depending on its position in the table, a word is jointly represented by two components: a row vector and a column vector. Since the words in the same row share the row vector and the words in the same column share the column vector, we only need 2 \sqrt(V) vectors to represent a vocabulary of |V| unique words, which are far less than the |V| vectors required by existing approaches. The LightRNN algorithm significantly reduces the model size and speeds up the training process, without sacrifice of accuracy (it achieves similar, if not better, perplexity as compared to state-of-the-art language models).


## Dependencies
- Python 2.7
- Tensorflow 0.12
- ortools 5.0.3919 (https://github.com/google/or-tools)

## Usage
python train_lm.py --data_path=./data/

## Others
- For minimum weight perfect matching algorithm, we use the minimum cost flow solver in ortools. 
- Currently, we test the code on PTB dataset.
- This is a initital version, I may optimize it in the future time. If you find any problems, please don't hesitate to contact me: eewangyisen AT gmail DOT com
