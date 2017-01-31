## Recurrent neural networks for Chinese named entity recognition in TensorFlow
This repository contains a neural network model for chainese named entity recognition.

## Contributer
- [Jingyuan Zhang](https://github.com/zjy-ucas)
- [Mingjie Chen](https://github.com/superthierry)

## Requirements
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [jieba](https://github.com/fxsjy/jieba)


## Model
The model is a birectional LSTM neural network with a CRF layer. Sequence of chinese characters are projected into sequence of dense vectors, and concated with extra features as the inputs of recurrent layer, here we employ one hot vectors representing word boundary features for illustration. The recurrent layer is a bidirectional LSTM layer, outputs of forward and backword vectors are concated and projected to score of each tag. A CRF layer is used to overcome label-bias problem.

## Basic Usage

### Default parameters:
- batch size: 20
- gradient clip: 5
- optimizer: Adam
- dropout rate: 0.5
- basic learning rate: 0.001

We train word vectors with gensim version of word2vec on Chinese WiKi corpus and 	finetune the embedding layer
#### Train the model with default parameters:
```shell
$ python3 train.py
```

the best F1 score is 0.8948 when tested, a f1 score higher than 0.89 is reasonable