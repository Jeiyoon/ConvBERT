# author: Jeiyoon
# This code is based on awesome works (See README.md)
from setproctitle import *

setproctitle('k4ke-convbert-test')

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import json
import argparse
import logging
import inspect
import random
from random import choice, shuffle, randrange
from typing import Callable, List, Optional, Set, Tuple, Union
import math

import torch
from torch import Tensor

import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback

logger = logging.getLogger()


def get_args(description="ConvBERT"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--task", default=None, type=str, required=True,
                        help="task name: pretrain or finetune")
    parser.add_argument("--vocab_model_path", default="web-crawler/kowiki/kowiki.model", type=str, required=False,
                        help="vocabulary model path (model)")
    parser.add_argument("--vocab_input_path", default="web-crawler/kowiki/kowiki.txt", type=str, required=False,
                        help="vocabulary input path (txt)")
    parser.add_argument("--vocab_output_path", default="vocab_bert/vocab_bert_{}.json", type=str, required=False,
                        help="vocabulary output path (json)")
    parser.add_argument("--finetune_data_path", default="naver_movie", type=str, required=False,
                        help="data path for finetuning")
    parser.add_argument("--pretrained_model_path", default="pretrain/save_convbert_pretrain.pth", type=str, required=False,
                        help="prtrained model path")

    args = parser.parse_args()

    return args


def generate_pretrain_data(vocab, count, n_seq, mask_prob):
    args = get_args()

    in_file = args.vocab_input_path  # pretrain 전처리에 사용할 corpus 경로
    out_file = args.vocab_output_path # 전처리후 저장되는 경로

    # generate pretrain dataset
    make_pretrain_data(vocab, in_file, out_file, count, n_seq, mask_prob)

def apply_chunking_to_forward(
    forward_fn: Callable[..., torch.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> torch.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.
    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.
    Args:
        forward_fn (`Callable[..., torch.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[torch.Tensor]`):
            The input tensors of `forward_fn` which will be chunked
    Returns:
        `torch.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.
    Examples:
    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    ```"""

    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return torch.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)


def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    pad_attn_mask = seq_k.data.eq(i_pad)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)

    return pad_attn_mask


def create_pretrain_mask(tokens, mask_cnt, vocab_list):
    cand_idx = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        """
        1. # u2581: 단어의 시작 (# cls -> u2581)
        - 이 값으로 시작하지 않으면 이전 token과 연결된 subword임
        """
        if 0 < len(cand_idx) and not token.startswith(u"\u2581"):
            cand_idx[-1].append(i)
        else:
            cand_idx.append([i])

    """
    2. 단어 랜덤 선택을 위해 단어 index를 섞음
    """
    shuffle(cand_idx)

    mask_lms = []

    rand = random.Random()

    for index_set in cand_idx:
        """
        3. mask_cnt는 전체 tokens 개수의 15%에 해당함
        """
        if len(mask_lms) >= mask_cnt:
            break
        if len(mask_lms) + len(index_set) > mask_cnt:
            continue
        for index in index_set:
            masked_token = None
            """
            4. index에 대해 80% 확률로 [MASK]로 치환
            """
            # if random() < 0.8: # 80% replace with [MASK]
            if rand.random() < 0.8:  # 80% replace with [MASK]
                masked_token = "[MASK]"
            else:
                """
                5. 남은 20%에 대해 10%는 현재 값을 유지, 10%는 vocab_list에서 임의의 값을 선택함
                """
                # if random() < 0.5: # 10% keep original
                if rand.random() < 0.5:  # 10% keep original
                    masked_token = tokens[index]
                else:  # 10% random word
                    masked_token = choice(vocab_list)
            """
            6. masked token의 index와 정답값을 저장 
            """
            mask_lms.append({"index": index, "label": tokens[index]})
            tokens[index] = masked_token

    mask_lms = sorted(mask_lms, key=lambda x: x["index"])
    mask_idx = [p["index"] for p in mask_lms]
    mask_label = [p["label"] for p in mask_lms]

    return tokens, mask_idx, mask_label


def trim_tokens(tokens_a, tokens_b, max_seq):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_seq:
            break

        """
        1. Tokens A의 길이가 길 경우 앞에서 부터 토큰을 제거
        2. Tokens B의 길이가 길 경우 뒤에서 부터 토큰을 제거
        """
        if len(tokens_a) > len(tokens_b):
            del tokens_a[0]
        else:
            tokens_b.pop()


def create_pretrain_instances(docs, doc_idx, doc, n_seq, mask_prob, vocab_list):
    # tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
    max_seq = n_seq - 3 # [CLS], [SEP], [SEP]
    tgt_seq = max_seq

    instances = []
    current_chunk = []
    current_length = 0

    for i in range(len(doc)):
        current_chunk.append(doc[i])  # line 추가
        current_length += len(doc[i]) # line의 token 수 추가

        """
        1. 마지막 줄이거나 target token 수를 넘을경우 학습 데이터 생성
        """
        if i == len(doc) - 1 or current_length >= tgt_seq:
            if 0 < len(current_chunk):
                a_end = 1
                if 1 < len(current_chunk):
                    # 랜덤하게 길이를 선택해서 tokens_a를 만들음
                    a_end = randrange(1, len(current_chunk))
                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []

                rand = random.Random()

                # if len(current_chunk) == 1 or random() < 0.5:
                if len(current_chunk) == 1 or rand.random() < 0.5:
                    is_next = 0 # False
                    tokens_b_len = tgt_seq - len(tokens_a)
                    random_doc_idx = doc_idx # 현재 doc idx

                    while doc_idx == random_doc_idx:
                        random_doc_idx = randrange(0, len(docs))
                    random_doc = docs[random_doc_idx] # 다른 doc index

                    random_start = randrange(0, len(random_doc))

                    for j in range(random_start, len(random_doc)):
                        tokens_b.extend(random_doc[j])
                else: # 0.5
                    is_next = 1 # True

                    for j in range(a_end, len(current_chunk)): # tokens_a에 이어서 tokens_b 생성
                        tokens_b.extend(current_chunk[j])

                """
                2. 토큰들의 길이가 최대길이를 넘지않게 줄이기 
                """
                trim_tokens(tokens_a, tokens_b, max_seq)

                assert 0 < len(tokens_a)
                assert 0 < len(tokens_b)

                tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
                segment = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1) # bert 학습을 위한 segment 나누기

                """
                3. Pretrain을 위한 [MASK] 생성
                """
                tokens, mask_idx, mask_label = create_pretrain_mask(tokens, int((len(tokens) - 3) * mask_prob),
                                                                    vocab_list)

                instance = {
                    "tokens": tokens,
                    "segment": segment,
                    "is_next": is_next, # A에 이어서 B를 생성했는지 (bool)
                    "mask_idx": mask_idx,
                    "mask_label": mask_label
                }
                instances.append(instance)

            current_chunk = []
            current_length = 0
    return instances


def make_pretrain_data(vocab, in_file, out_file, count, n_seq, mask_prob):
    # len(vocab_list): 8006
    """
    1. unknown 제거
    """
    vocab_list = []
    for id in range(vocab.get_piece_size()):
        if not vocab.is_unknown(id):
            vocab_list.append(vocab.id_to_piece(id))

    """
    2. 학습에 사용할 말뭉치가 총 몇 라인인지 확인
    """
    # line_cnt: 4875507
    line_cnt = 0
    with open(in_file, "r") as in_f:
        for _ in in_f:
            line_cnt += 1

    """
    3. 각 라인마다 vocab을 이용해 tokenize한 뒤 docs 리스트에 추가
    """
    # list(623167)
    docs = [] # 단락 배열
    with open(in_file, "r") as f:
        doc = [] # 단락
        with tqdm(total=line_cnt, desc=f"Loading") as pbar:
            for i, line in enumerate(f):
                line = line.strip()
                if line == "":
                    if 0 < len(doc): # 빈 줄이면 단락의 끝이므로 doc을 docs에 추가하고 doc을 새로 만들음
                        docs.append(doc)
                        doc = []
                else:
                    pieces = vocab.encode_as_pieces(line)
                    if 0 < len(pieces):
                        doc.append(pieces)
                pbar.update(1)
        if doc:
            docs.append(doc)

    """
    4. count 횟수 만큼 돌면서 pretrain data를 생성함
    - 왜냐하면 bert는 Masking을 15%만 하기 때문에 한번에 전체적인 단어를 학습할 수 가 없음
    - e.g., count = 10
    """
    for index in range(count):
        output = out_file.format(index)
        if os.path.isfile(output): continue

        with open(output, "w") as out_f:
            with tqdm(total=len(docs), desc=f"Making") as pbar:
                for i, doc in enumerate(docs):
                    """
                    5. 단락(doc)별 pretrain 데이터 생성 
                    """
                    instances = create_pretrain_instances(docs, i, doc, n_seq, mask_prob, vocab_list)
                    for instance in instances:
                        out_f.write(json.dumps(instance))
                        out_f.write("\n")
                    pbar.update(1)


def pretrin_collate_fn(inputs):
    labels_cls, labels_lm, inputs, segments = list(zip(*inputs))

    labels_lm = torch.nn.utils.rnn.pad_sequence(labels_lm, batch_first=True, padding_value=-1)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    segments = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True, padding_value=0)

    batch = [
        torch.stack(labels_cls, dim=0),
        labels_lm,
        inputs,
        segments
    ]
    return batch


def finetune_collate_fn(inputs):
    labels, inputs, segments = list(zip(*inputs))

    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    segments = torch.nn.utils.rnn.pad_sequence(segments, batch_first=True, padding_value=0)

    batch = [
        torch.stack(labels, dim=0),
        inputs,
        segments,
    ]
    return batch


def train_epoch(config, epoch, model, criterion_lm, criterion_cls, optimizer, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
        for i, value in enumerate(train_loader):
            labels_cls, labels_lm, inputs, segments = map(lambda v: v.to(config.device), value)

            optimizer.zero_grad()
            outputs = model(inputs, segments)
            logits_cls, logits_lm = outputs[0], outputs[1]

            loss_cls = criterion_cls(logits_cls, labels_cls)
            loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))
            loss = loss_cls + loss_lm

            loss_val = loss_lm.item()
            losses.append(loss_val)

            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)

# finetune_epoch(config, epoch, model, criterion_cls, optimizer, train_loader)
# criterion_cls = torch.nn.CrossEntropyLoss()
def finetune_epoch(config, epoch, model, criterion_cls, optimizer, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train({epoch})") as pbar:
        for i, value in enumerate(train_loader):
            labels, inputs, segments = map(lambda v: v.to(config.device), value)

            optimizer.zero_grad()
            outputs = model(inputs, segments)
            # logits_cls = outputs[0]
            # logits_cls = outputs[:, 0]
            logits_cls = torch.argmax(outputs, dim=1)

            # logits_cls: Tensor(2,) -> Tensor(32, )
            # labels: Tensor(32, )
            loss_cls = criterion_cls(logits_cls.float(), labels.float())
            loss = loss_cls

            loss_val = loss_cls.item()
            losses.append(loss_val)

            loss.requires_grad_(True)

            loss.backward()
            optimizer.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)


def eval_epoch(config, model, data_loader):
    matchs = []
    model.eval()

    n_word_total = 0
    n_correct_total = 0
    with tqdm(total=len(data_loader), desc=f"Valid") as pbar:
        for i, value in enumerate(data_loader):
            labels, inputs, segments = map(lambda v: v.to(config.device), value)

            outputs = model(inputs, segments)
            logits_cls = outputs[0]
            _, indices = logits_cls.max(1)

            match = torch.eq(indices, labels).detach()
            matchs.extend(match.cpu())
            accuracy = np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0

            pbar.update(1)
            pbar.set_postfix_str(f"Acc: {accuracy:.3f}")
    return np.sum(matchs) / len(matchs) if 0 < len(matchs) else 0


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    acc = accuracy_score(labels, preds)

    return acc


def finetune(model, config, train_loader, test_loader):
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(config.device)

    learning_rate = 5e-5
    n_epoch = 10

    trainer = Trainer(max_epochs=n_epoch, gpus=4, devices=config.device)
    trainer.fit(model, train_dataloaders=train_loader)
    trainer.test(test_dataloaders=test_loader)

    criterion_cls = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_epoch, best_loss, best_score = 0, 0, 0
    losses, scores = [], []

    for epoch in range(n_epoch):
        # loss = finetune_epoch(config, epoch, model, criterion_cls, optimizer, train_loader)
        score = eval_epoch(config, model, test_loader)

        # losses.append(loss)
        scores.append(score)

        if best_score < score:
            # best_epoch, best_loss, best_score = epoch, loss, score
            best_epoch, best_score = epoch, score
    print(f">>>> epoch={best_epoch}, loss={best_loss:.5f}, socre={best_score:.5f}")
    return losses, scores


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)


class SeparableConv1D(nn.Module):
    def __init__(self, config, input_filters, output_filters, kernel_size, **kwargs):
        super().__init__()

        # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        self.depthwise = nn.Conv1d(
            input_filters,
            input_filters,
            kernel_size=kernel_size,
            groups=input_filters,
            padding=kernel_size // 2,
            bias=False,
        )
        self.pointwise = nn.Conv1d(input_filters, output_filters, kernel_size=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_filters, 1))

        self.depthwise.weight.data.normal_(mean=0.0, std=config.initializer_range)
        self.pointwise.weight.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(hidden_states)
        x = self.pointwise(x)
        x += self.bias

        return x


class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class SpanBasedDynamicConvolution(nn.Module):
    def __init__(self, config):
        super().__init__()
        """
        Bottleneck design for self-attention
        """
        # e.g., 12 // 2 = 6
        new_num_attention_heads = config.n_head // config.head_ratio

        if new_num_attention_heads < 1:  # head ratio exception
            self.head_ratio = config.n_head
            self.num_attention_heads = 1
        else:
            self.num_attention_heads = new_num_attention_heads  # new size
            self.head_ratio = config.head_ratio

        self.conv_kernel_size = config.conv_kernel_size  # e.g., 9

        if config.d_hidn % self.num_attention_heads != 0:
            raise ValueError("hidden_size should be divisible by num_attention_heads")

        self.attention_head_size = config.d_hidn // config.n_head  # e.g., 768 // 12 = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # e.g., 6 * 64 = 384

        # self.W_Q = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head) # 12 * 64
        self.W_Q = nn.Linear(config.d_hidn, self.all_head_size)  # e.g., 6 * 64 = 384
        self.W_K = nn.Linear(config.d_hidn, self.all_head_size)
        self.W_V = nn.Linear(config.d_hidn, self.all_head_size)

        # def __init__(self, config, input_filters, output_filters, kernel_size, **kwargs):
        self.key_conv_attn_layer = SeparableConv1D(
            config, config.d_hidn, self.all_head_size, self.conv_kernel_size
        )
        self.conv_kernel_layer = nn.Linear(self.all_head_size,
                                           self.num_attention_heads * self.conv_kernel_size)  # Linear(384, 6 * 9)

        self.conv_out_layer = nn.Linear(config.d_hidn, self.all_head_size)

        # https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
        # https://npclinic3.tistory.com/6
        self.unfold = nn.Unfold(
            kernel_size=[self.conv_kernel_size, 1],
            padding=[int((self.conv_kernel_size - 1) / 2), 0]
        )

        self.dropout = nn.Dropout(config.dropout)
        # self.scaled_dot_attn = ScaledDotProductAttention(self.config) # self-attn

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = False,
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mixed_query_layer = self.W_Q(hidden_states)
        batch_size = hidden_states.size(0)

        if encoder_hidden_states is not None:
            mixed_key_layer = self.W_K(encoder_hidden_states)
            mixed_value_layer = self.W_V(encoder_hidden_states)
        else:
            mixed_key_layer = self.W_K(hidden_states)
            mixed_value_layer = self.W_V(hidden_states)

        mixed_key_conv_attn_layer = self.key_conv_attn_layer(hidden_states.transpose(1, 2))
        mixed_key_conv_attn_layer = mixed_key_conv_attn_layer.transpose(1, 2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)

        """
        [Multiply]
        """
        conv_attn_layer = torch.multiply(mixed_key_conv_attn_layer, mixed_query_layer)

        """
        [Linear]        
        """
        conv_kernel_layer = self.conv_kernel_layer(conv_attn_layer)
        conv_kernel_layer = torch.reshape(conv_kernel_layer, [-1, self.conv_kernel_size, 1])

        """
        [Softmax]
        """
        conv_kernel_layer = torch.softmax(conv_kernel_layer, dim=1)

        """
        [LConv & Value multiplication]
        """
        conv_out_layer = self.conv_out_layer(hidden_states)
        conv_out_layer = torch.reshape(conv_out_layer, [batch_size, -1, self.all_head_size])
        conv_out_layer = conv_out_layer.transpose(1, 2).contiguous().unsqueeze(-1)
        conv_out_layer = nn.functional.unfold(
            conv_out_layer,
            kernel_size=[self.conv_kernel_size, 1],
            dilation=1,
            padding=[(self.conv_kernel_size - 1) // 2, 0],
            stride=1,
        )
        conv_out_layer = conv_out_layer.transpose(1, 2).reshape(
            batch_size, -1, self.all_head_size, self.conv_kernel_size
        )
        conv_out_layer = torch.reshape(conv_out_layer, [-1, self.attention_head_size, self.conv_kernel_size])

        conv_out_layer = torch.matmul(conv_out_layer, conv_kernel_layer)
        conv_out_layer = torch.reshape(conv_out_layer, [-1, self.all_head_size])

        """
        Self-attention
        """
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # scale

        # attention_scores: Tensor(128, 6, 256, 256)
        # attention_mask: Tensor(128, 256, 256) -> Tensor(128, 6, 256, 256)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # softmax
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # value
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # context_layer: Self-Attn
        # conv_out: SDConv
        conv_out = torch.reshape(conv_out_layer, [batch_size, -1, self.num_attention_heads, self.attention_head_size])

        """
        Mixed Attention ([Concat])
        """
        context_layer = torch.cat([context_layer, conv_out], 2)

        # context_layer: Tensor(128, 256, 12, 64)
        # context_layer.size()[:-2]: torch.Size([128, 256])
        # self.head_ratio: 2
        # self.all_head_size: 384
        # new_context_layer_shape: (128, 256, 768)
        new_context_layer_shape = context_layer.size()[:-2] + (self.head_ratio * self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class ConvBertSelfOutput(nn.Module):
    def __init__(self, config):
        """
        BERT output을 만들기 위한 layer들 정의
        """
        super().__init__()

        self.dense = nn.Linear(config.d_hidn, config.d_hidn)
        self.LayerNorm = nn.LayerNorm(config.d_hidn, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,
                hidden_states: torch.Tensor,
                input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class ConvBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SpanBasedDynamicConvolution(config)
        self.output = ConvBertSelfOutput(config)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                encoder_hidden_states: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = False) -> Tuple[torch.Tensor, Optional[torch.FloatTensor]]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            output_attentions,
        )

        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]

        return outputs


class GroupedLinearLayer(nn.Module):
    def __init__(self, input_size, output_size, num_groups):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.num_groups = num_groups
        self.group_in_dim = self.input_size // self.num_groups
        self.group_out_dim = self.output_size // self.num_groups

        self.weight = nn.Parameter(torch.empty(self.num_groups, self.group_in_dim, self.group_out_dim))
        self.bias = nn.Parameter(torch.empty(output_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = list(hidden_states.size())[0]
        x = torch.reshape(hidden_states, [-1, self.num_groups, self.group_in_dim])
        x = x.permute(1, 0, 2)
        x = torch.matmul(x, self.weight)
        x = x.permute(1, 0, 2)
        x = torch.reshape(x, [batch_size, -1, self.output_size])
        x = x + self.bias

        return x


class ConvBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.num_groups == 1:
            self.dense = nn.Linear(config.d_hidn, config.intermediate_size)
        else:
            self.dense = GroupedLinearLayer(
                input_size=config.d_hidn,
                output_size=config.intermediate_size,
                num_groups=config.num_groups
            )

        self.intermediate_act_fn = GELUActivation()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ConvBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.num_groups == 1:
            self.dense = nn.Linear(config.intermediate_size, config.d_hidn)
        else:
            self.dense = GroupedLinearLayer(
                input_size=config.intermediate_size,
                output_size=config.d_hidn,
                num_groups=config.num_groups
            )

        self.LayerNorm = nn.LayerNorm(config.d_hidn, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # https://huggingface.co/docs/transformers/main_classes/configuration#configuration
        self.chunk_size_feedforward = 0  # config.chunk_size_feed_forward: No linear separation
        self.seq_len_dim = 1
        self.attention = ConvBertAttention(config)
        self.is_decoder = False  # config.is_decoder
        self.add_cross_attention = False  # config.add_cross_attention

        self.intermediate = ConvBertIntermediate(config)
        self.output = ConvBertOutput(config)

    # def forward(self, inputs, attn_mask):
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.FloatTensor]]:
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # Tensor(128, 256, 768)
        return outputs # attention_output

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 12 (based-size model)
        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])
        self.gradient_checkpointing = False

        self.enc_emb = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidn)
        self.pos_emb = nn.Embedding(self.config.n_enc_seq + 1, self.config.d_hidn)
        self.seg_emb = nn.Embedding(self.config.n_seg_type, self.config.d_hidn)  # to distinguish different sentences

    def forward(self, inputs, segments):
        positions = torch.arange(inputs.size(1),
                                 device=inputs.device,
                                 dtype=inputs.dtype).expand(inputs.size(0), inputs.size(1)).contiguous() + 1

        pos_mask = inputs.eq(self.config.i_pad)
        positions.masked_fill_(pos_mask, 0)

        # (bs, n_enc_seq, d_hidn)
        outputs = self.enc_emb(inputs) + self.pos_emb(positions) + self.seg_emb(segments)

        # (bs, n_enc_seq, n_enc_seq)
        attn_mask = get_attn_pad_mask(inputs, inputs, self.config.i_pad)

        # self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])
        for layer in self.layers:
            # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
            outputs = layer(outputs, attn_mask)

        # (bs, n_enc_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return outputs


class ConvBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        ConvBERT Encoder 선언 부분 
        """
        self.encoder = Encoder(self.config)

        self.linear = nn.Linear(config.d_hidn, config.d_hidn)
        self.activation = torch.tanh

    def forward(self, inputs, segments):
        # (bs, n_seq, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        # outputs, self_attn_probs = self.encoder(inputs, segments)
        outputs = self.encoder(inputs, segments)
        # (bs, d_hidn)
        outputs_cls = outputs[:, 0].contiguous()
        outputs_cls = self.linear(outputs_cls)
        outputs_cls = self.activation(outputs_cls)

        # (bs, n_enc_seq, n_enc_vocab), (bs, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return outputs, outputs_cls  # , self_attn_probs

    def save(self, epoch, loss, path):
        torch.save({
            "epoch": epoch,
            "loss": loss,
            "state_dict": self.state_dict()
        }, path)

    def load(self, path):
        save = torch.load(path)
        self.load_state_dict(save["state_dict"])

        return save["epoch"], save["loss"]


class BERTPretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        """
        ConvBERT 모델을 가져와서 학습
        """
        self.bert = ConvBERT(self.config)
        # nsp: classfier objective
        self.projection_cls = nn.Linear(self.config.d_hidn, 2, bias=False)
        # mlm: lm objective
        self.projection_lm = nn.Linear(self.config.d_hidn, self.config.n_enc_vocab, bias=False)
        self.projection_lm.weight = self.bert.encoder.enc_emb.weight  # weight sharing

    def forward(self, inputs, segments):
        """
        두가지 task인 nsp와 mlm에 대해 학습을 진행함
        """
        # (bs, n_enc_seq, d_hidn), (bs, d_hidn), [(bs, n_head, n_enc_seq, n_enc_seq)]
        outputs, outputs_cls = self.bert(inputs, segments)
        # (bs, 2)
        logits_cls = self.projection_cls(outputs_cls)
        # (bs, n_enc_seq, n_enc_vocab)
        logits_lm = self.projection_lm(outputs)
        # (bs, n_enc_vocab), (bs, n_enc_seq, n_enc_vocab), [(bs, n_head, n_enc_seq, n_enc_seq)]
        return logits_cls, logits_lm  # , attn_probs


class PretrainDataSet(torch.utils.data.Dataset):
    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.labels_cls = []
        self.labels_lm = []
        self.sentences = []
        self.segments = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc=f"Loading {infile}", unit=" lines")):
                instance = json.loads(line)
                self.labels_cls.append(instance["is_next"])
                sentences = [vocab.piece_to_id(p) for p in instance["tokens"]]
                self.sentences.append(sentences)
                self.segments.append(instance["segment"])
                mask_idx = np.array(instance["mask_idx"], dtype=np.int)
                mask_label = np.array([vocab.piece_to_id(p) for p in instance["mask_label"]], dtype=np.int)
                label_lm = np.full(len(sentences), dtype=np.int, fill_value=-1)
                label_lm[mask_idx] = mask_label
                self.labels_lm.append(label_lm)

    def __len__(self):
        assert len(self.labels_cls) == len(self.labels_lm)
        assert len(self.labels_cls) == len(self.sentences)
        assert len(self.labels_cls) == len(self.segments)
        return len(self.labels_cls)

    def __getitem__(self, item):
        return (torch.tensor(self.labels_cls[item]),
                torch.tensor(self.labels_lm[item]),
                torch.tensor(self.sentences[item]),
                torch.tensor(self.segments[item]))


class BERTFineTune(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert = ConvBERT(self.config)

        # classification task
        self.projection_cls = nn.Linear(self.config.d_hidn, self.config.n_output, bias=False)

    def forward(self, inputs, segments):
        outputs, outputs_cls = self.bert(inputs, segments)

        logit_cls = self.projection_cls(outputs_cls)

        # logit_cls = torch.argmax(logit_cls, dim=1)

        return logit_cls


class FineTuneDataSet(torch.utils.data.Dataset):
    def __init__(self, vocab, infile):
        self.vocab = vocab
        self.labels = []
        self.sentences = []
        self.segments = []

        line_cnt = 0
        with open(infile, "r") as f:
            for line in f:
                line_cnt += 1

        with open(infile, "r") as f:
            for i, line in enumerate(tqdm(f, total=line_cnt, desc="Loading Dataset", unit=" lines")):
                data = json.loads(line)
                self.labels.append(data["label"])
                sentence = [vocab.piece_to_id("[CLS]")] + [vocab.piece_to_id(p) for p in data["doc"]] + [
                    vocab.piece_to_id("[SEP]")]
                self.sentences.append(sentence)
                self.segments.append([0] * len(sentence))

    def __len__(self):
        assert len(self.labels) == len(self.sentences)
        assert len(self.labels) == len(self.segments)
        return len(self.labels)

    def __getitem__(self, item):
        return (torch.tensor(self.labels[item]),
                torch.tensor(self.sentences[item]),
                torch.tensor(self.segments[item]))


def main():
    """
    1. 코드 동작에 필요한 argument들 불러오기
    """
    args = get_args()

    """
    2. kowiki를 활용한 vocab model 불러오기
    - vocab size = 8000
    """
    vocab_path = args.vocab_model_path
    vocab = spm.SentencePieceProcessor()
    vocab.load(vocab_path)

    """
    3. ConvBERT model configuration 설정
    """
    config = Config({
        "n_enc_vocab": len(vocab),
        "n_enc_seq": 512,
        "n_seg_type": 2,
        "n_layer": 12,
        "d_hidn": 768,
        "intermediate_size": 3072,
        "i_pad": 0,
        "d_ff": 3072,
        "n_head": 12,
        "d_head": 64,
        "dropout": 0.1,
        "layer_norm_epsilon": 1e-12,
        "n_output": 2,
        "head_ratio": 2,
        "initializer_range": 0.02,
        "conv_kernel_size": 9,
        "num_groups": 1,
    })
    task = args.task # pretrain 또는 finetune

    learning_rate = 5e-5
    n_epoch = 20
    count = 10  # # of corpus
    n_seq = 256
    mask_prob = 0.15
    batch_size = 32

    if task == "pretrain":
        """
        4. corpus를 활용한 pretrain dataset 인스턴스 생성 후 json으로 저장
        """
        generate_pretrain_data(vocab, count, n_seq, mask_prob)

        dataset = PretrainDataSet(vocab, f"vocab_bert/vocab_bert_0.json")
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   collate_fn=pretrin_collate_fn)

        config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """
        5. Pretrain
        """
        model = BERTPretrain(config)

        save_pretrain = "pretrain/save_convbert_pretrain.pth"
        best_epoch, best_loss = 0, 0

        if os.path.isfile(save_pretrain):
            best_epoch, best_loss = model.bert.load(save_pretrain)
            print(f"load pretrain from: {save_pretrain}, epoch={best_epoch}, loss={best_loss}")
            best_epoch += 1

        model.to(config.device)

        criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        criterion_cls = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        losses = []
        offset = best_epoch
        for step in range(n_epoch):
            epoch = step + offset
            if 0 < step:
                del train_loader
                dataset = PretrainDataSet(vocab, f"vocab_bert/vocab_bert_{epoch % count}.json")
                train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                                           collate_fn=pretrin_collate_fn)

            """
            Train
            """
            loss = train_epoch(config, epoch, model, criterion_lm, criterion_cls, optimizer, train_loader)
            losses.append(loss)
            model.bert.save(epoch, loss, save_pretrain)

        data = {
            "loss": losses
        }
    elif task == "finetune":
        model_name = "convbert_ft"
        learning_rate = 2e-5
        n_epoch = 10
        batch_size = 32
        wd = 0.01

        data_dir = args.finetune_data_path

        train_dataset = FineTuneDataSet(vocab, f"{data_dir}/ratings_train.json")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                   collate_fn=finetune_collate_fn)
        test_dataset = FineTuneDataSet(vocab, f"{data_dir}/ratings_test.json")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                                  collate_fn=finetune_collate_fn)

        # loading
        model = BERTFineTune(config)
        save_pretrain_path = args.pretrained_model_path
        model.bert.load(save_pretrain_path)

        args_ft =TrainingArguments(
            f"test-{model_name}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=n_epoch,
            seed=42,
            weight_decay=wd,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model,
            args_ft,
            train_dataset=train_loader,
            eval_dataset=test_loader,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()

    else:
        raise ValueError("task name should be pretrain or finetune")


if __name__ == "__main__":
    main()
