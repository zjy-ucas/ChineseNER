## Recurrent neural networks for Chinese named entity recognition in TensorFlow
This repository contains a neural network model for chainese named entity recognition.

## Contributer
- [Jingyuan Zhang](https://github.com/zjy-ucas)
- [Mingjie Chen](https://github.com/superthierry)

## Requirements
- [Tensorflow=1.2.0](https://github.com/tensorflow/tensorflow)
- [jieba](https://github.com/fxsjy/jieba)


## Model
The model is a birectional LSTM neural network with a CRF layer. Sequence of chinese characters are projected into sequence of dense vectors, and concated with extra features as the inputs of recurrent layer, here we employ one hot vectors representing word boundary features for illustration. The recurrent layer is a bidirectional LSTM layer, outputs of forward and backword vectors are concated and projected to score of each tag. A CRF layer is used to overcome label-bias problem.

## Basic Usage

### Default parameters:
- batch size: 1
- gradient clip: 5
- optimizer: SGD
- dropout rate: 0.5
- learning rate: 0.005

We train word vectors with gensim version of word2vec on Chinese WiKi corpus
#### Train the model with default parameters:
```shell
$ python3 main.py --train=True --clean=True
```

#### Online evaluate:
```shell
$ python3 main.py
```


