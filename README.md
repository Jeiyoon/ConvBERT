# Description

- This repository reimplemented "[ConvBERT: Improving BERT with Span-based Dynamic Convolution (Jiang et al., NeurIPS 2020)](https://arxiv.org/pdf/2008.02496.pdf)" based on awesome works (See Reference).

# Requirements

1. python=3.7
2. pytorch==1.12.1
3. sentencepiece==0.1.97
4. setproctitle

# Dataset

## Pretrain: Kowiki (Korean)

- Download
~~~
$ cd web-crawler
$ pip install tqdm
$ pip install pandas
$ pip install bs4
$ pip install wget
$ pip install pymongo
$ python kowiki.py
~~~

- Result 
```
['▁겨울', '이', '▁되어', '서', '▁날', '씨', '가', '▁무', '척', '▁추', '워', '요', '.']
[3217, 3625, 677, 3639, 715, 4073, 3636, 106, 4227, 197, 3993, 3796, 3627]

이번 성탄절은 화이트 크리스마스가 될까요?
['▁이번', '▁성', '탄', '절', '은', '▁화', '이트', '▁크리스', '마', '스가', '▁될', '까', '요', '?']
[2894, 89, 4006, 3961, 3641, 269, 649, 1895, 3698, 711, 1453, 3834, 3796, 4308]

겨울에 감기 조심하시고 행복한 연말 되세요.
['▁겨울', '에', '▁감', '기', '▁조', '심', '하', '시', '고', '▁행', '복', '한', '▁연', '말', '▁되', '세', '요', '.']
[3217, 3628, 193, 3643, 53, 3872, 3633, 3650, 3638, 246, 3911, 3640, 63, 3869, 472, 3719, 3796, 3627]
```

## Finetune: Naver sentiment movie corpus v1.0 (Korean)

- Download
~~~
$ cd web-crawler
$ wget https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt
$ wget https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt
$ python naver_movie.py
~~~
