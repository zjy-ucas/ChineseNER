## Recurrent neural networks for Chinese named entity recognition in TensorFlow
This repository contains a simple demo for chainese named entity recognition.

## Contributer
- [Jingyuan Zhang](https://github.com/zjy-ucas)
- [Mingjie Chen](https://github.com/superthierry)
- some data processing codes from [glample/tagger](https://github.com/glample/tagger)


## Requirements
- [Tensorflow=1.2.0](https://github.com/tensorflow/tensorflow)
- [jieba=0.37](https://github.com/fxsjy/jieba)


## Model
The model is a birectional LSTM neural network with a CRF layer. Sequence of chinese characters are projected into sequence of dense vectors, and concated with extra features as the inputs of recurrent layer, here we employ one hot vectors representing word boundary features for illustration. The recurrent layer is a bidirectional LSTM layer, outputs of forward and backword vectors are concated and projected to score of each tag. A CRF layer is used to overcome label-bias problem.

Our model is similar to the state-of-the-art Chinese named entity recognition model proposed in Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition.

## Basic Usage

### Default parameters:
- batch size: 20
- gradient clip: 5
- embedding size: 100
- optimizer: Adam
- dropout rate: 0.5
- learning rate: 0.001

Word vectors are trained with gensim version of word2vec on Chinese WiKi corpus, provided by [Chuanhai Dong](https://github.com/sea2603).

### Train the model with default parameters:
```shell
$ python3 main.py --train=True --clean=True
```

### Online evaluate:
```shell
$ python3 main.py
```

## Suggested readings:
1. [Natural Language Processing (Almost) from Scratch](http://jmlr.org/papers/volume12/collobert11a/collobert11a.pdf).  
Propose a unified neural network architecture for sequence labeling tasks.
2. [Neural Architectures for Named Entity Recognition](http://arxiv.org/abs/1603.01360).  
[End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](http://www.cs.cmu.edu/~xuezhem/publications/lstm-cnn-crf.pdf).  
Combine Character-based word representations and word representations to enhance sequence labeling systems.
3. [Transfer Learning for Sequence Tagging with Hierarchical Recurrent Networks](http://www.cs.cmu.edu/~./wcohen/postscript/iclr-2017-transfer.pdf).  
[Multi-task Multi-domain Representation Learning for Sequence Tagging](http://xueshu.baidu.com/s?wd=paperuri%3A%288d2ae013d4ea38b3aba07a5f5cf8c8d1%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Farxiv.org%2Fpdf%2F1608.02689v1.pdf&ie=utf-8&sc_us=16810667041741374202).  
Transfer learning for sequence tagging.
4. [Named Entity Recognition for Chinese Social Media with Jointly Trained Embeddings](http://www.aclweb.org/website/anthology/D/D15/D15-1064.pdf).  
Propose a joint training objective for the embeddings that makes use of both (NER) labeled and unlabeled raw text
5. [Improving Named Entity Recognition for Chinese Social Media with Word Segmentation Representation Learning](http://anthology.aclweb.org/P/P16/P16-2025.pdf).  
[An Empirical Study of Automatic Chinese Word Segentation for Spoken Language Understanding and Named Entity Recognition](http://www.aclweb.org/anthology/N/N16/N16-1028.pdf).  
Using word segmentation outputs as additional features for sequence labeling syatems.
6. [Semi-supervised Sequence Tagging with Bidirectional Language Models](http://xueshu.baidu.com/s?wd=paperuri%3A%28e7dcf1a507dabc77f1e26c28068ca937%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Farxiv.org%2Fpdf%2F1705.0108&ie=utf-8&sc_us=17831018953161676191).  
State-of-the-art model on Conll03 NER task, adding pre-trained context embeddings from bidirectional language models for sequence labeling task.
7. [Character-Based LSTM-CRF with Radical-Level Features for Chinese Named Entity Recognition](http://tcci.ccf.org.cn/conference/2016/papers/119.pdf).  
State-of-the-art model on SIGHAN2006 NER task.
8. [Named Entity Recognition with Bidirectional LSTM-CNNs](http://xueshu.baidu.com/s?wd=paperuri%3A%28995499661ccaa95ca3688318f4bc594b%29&filter=sc_long_sign&tn=SE_xueshusource_2kduw22v&sc_vurl=http%3A%2F%2Farxiv.org%2Fabs%2F1511.08308&ie=utf-8&sc_us=14130444594064699095).  
Method to apply lexicon features.

实现架构：
1. 读取数据集，数据集共三个文件，训练集，交叉测试集和测试集，文件中每一行包含两个元素，字和标识。每一句话间由一个空格隔开
 
            
2. 处理数据集
    1） 更新数据集中的标签，如： 单独的B-LOC→S-LOC，B-LOC,I-LOC→B-LOC,E-LOC，B-LOC,I-LOC,I-LOC→B-LOC, I-LOC, E-LOC
    2） 给每个char和tag分配一个id，得到一个包含所有字的字典dict，以及char_to_id, id_to_char, tag_to_id, id_to_tag, 将其存在map.pkl中
3. 准备训练集
        将训练集中的每句话变成4个list，第一个list是字，如[今，天，去，北，京]，第二个list是char_to_id [3,5,6,8,9]，第三个list是通过jieba分词得到的分词信息特征，如[1,3,0,1,3] （1，词的开始，2，词的中间，3，词的结尾，0，单个词），第四个list是target，如[0,0,0,2,3](非0的元素对应着tag_to_id中的数值)
4. BatchManager 将训练集划分成若干个batch，每个batch有20个句子，划分时，是现按句子长度从大到小排列
5. 配置model的参数
6. 构建模型
    1）input： 输入两个特征，char_to_id的list以及通过jieba得到的分词特征list
    2）embedding: 预先训练好了100维词向量模型，通过查询将得到每个字的100维向量，加上分词特征向量，输出到drouput(0.5)
    3）bi-lstm
    4）project_layer：两层的Wx+b  逻辑回归
    5）loss_layer：内嵌了CRF
 
