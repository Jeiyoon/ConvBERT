# Description

- This repository reimplemented "[ConvBERT: Improving BERT with Span-based Dynamic Convolution (Jiang et al., NeurIPS 2020)](https://arxiv.org/pdf/2008.02496.pdf)" based on awesome works (See Reference).

# Requirements

1. python=3.7
2. pytorch==1.12.1
3. sentencepiece==0.1.97
4. tqdm, setproctitle

# Dataset

## Pretrain: Kowiki (Korean)

~~~
$ git clone https://github.com/paul-hyun/web-crawler.git
$ cd web-crawler
$ pip install tqdm
$ pip install pandas
$ pip install bs4
$ pip install wget
$ pip install pymongo
$ python kowiki.py
~~~
