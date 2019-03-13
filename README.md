# Bidirectional LSTM: Abstract Text Summarization
## 双方向LSTMを用いた文脈を捉える抽象型文章要約
![](http://abigailsee.com/img/pointer-gen.png)

## Introduction

Extraction type is an approach of extracting a sentence that seems to be important from sentences to be summarized and creating a summary. The advantages and disadvantages are as follows.        

- Pros: Select a sentence in the original sentence to create a summary, so it is less likely to be a summary that is completely out of the way, and it will not be a grammatically strange summary       

- Cons: Because you can not use words that are not in the sentence, you can not use abstractions, paraphrases, or conjunctions to make them easier to read. Because of this, the summary created is a crude impression.  

I built seq2seq Bidirectional LSTM for Text Summarization task. Also LSTM with Attention is major method for summarization.               

## Technical Preferences

| Title | Detail |
|:-----------:|:------------------------------------------------|
| Environment | MacOS Mojave 10.14.3 |
| Language | Python |
| Library | Kras, scikit-learn, Numpy, matplotlib, Pandas, Seaborn |
| Dataset | [BBC Datasets](http://mlg.ucd.ie/datasets/bbc.html) |
| Algorithm | Encoder-Decoder LSTM |

## Refference

- [Get To The Point: Summarization with Pointer-Generator Networks](https://nlp.stanford.edu/pubs/see2017get.pdf)
- [Bidirectional Attentional Encoder-Decoder Model and Bidirectional Beam Search for Abstractive Summarization](https://arxiv.org/pdf/1809.06662.pdf)
- [Taming Recurrent Neural Networks for Better Summarization](http://www.abigailsee.com/2017/04/16/taming-rnns-for-better-summarization.html)
- [Encoder-Decoder Models for Text Summarization in Keras](https://machinelearningmastery.com/encoder-decoder-models-text-summarization-keras/)
- [Text Summarization Using Keras Models](https://hackernoon.com/text-summarization-using-keras-models-366b002408d9)
- [Text Summarization in Python: Extractive vs. Abstractive techniques revisited](https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/)
- [大自然言語時代のための、文章要約](https://qiita.com/icoxfog417/items/d06651db10e27220c819)
