
import glob
import re
import os
import unicodedata
import string
import sys
import codecs
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from io import open
from sklearn.metrics import f1_score
from sklearn import datasets, linear_model
from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
import matplotlib

def batched_index_select(input, dim, index):
    #print(input[0].shape)
    #print(index.shape)
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    #print(index.shape)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    """
    print ("index ")
    print (index[1])
    print ("input ")
    print (input)
    print(torch.gather(input, dim, index)[1])
    print(torch.gather(input, dim, index).shape)
"""
    return torch.gather(input, dim, index)


def convert_tokens_to_features(examples, max_seq_length,  tokenizer ):

    features = []
   # print ("label_map :",label_map )
    #print ("label_list :",label_list )
    #print ( len ( examples))
    example_counter=[ ]

    for (ex_index, x) in enumerate(examples):
        text_b ,text_a = x
        #tokens_a = tokenizer.tokenize( text_a)
        tokens_a = text_a.split( )
        #tokens_a  =  [w for w in tokens_a if not w in stop_words]
        example_counter.append(len(tokens_a)  )
        #tokens_b = " ".join(eval( text_b )).split( )
        #tokens_b =  eval( text_b )
        tokens_b = " ".join(eval( text_b )).split( )
        #text_b:
        #tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        #else:
        #    # Account for [CLS] and [SEP] with "- 2"
        #3    if len(tokens_a) > max_seq_length - 2:
        #        tokens_a = tokens_a[:(max_seq_length - 2)]
        #print ("example")
        #print (example.text_a  )
        #print ("example tokens ")
        #print (tokens_a )

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        f_dict = {
            "input_ids" :input_ids,
            "input_mask" :input_mask,
            "type_ids" : segment_ids
        }
        #features.append( input_ids,input_mask,segment_ids )
        features.append(f_dict )


    return features


#BERTに入れるための入力に変換
def convert_full_sequence_sdp_to_features(examples, max_seq_length,  tokenizer ):

    features = []
    #print ("label_map :",label_map )
    #print ("label_list :",label_list )
    #print ( len ( examples))
    #print(tokenizer)
    example_counter=[ ]

    for (ex_index, x) in enumerate(examples):
        #text_b ,text_a = x
        text_a ,sdp_postion= x #文とsdpのリストに分ける
        #print(sdp_postion)
        #tokens_a = tokenizer.tokenize( text_a)
        tokens_a = text_a.split( )#文を単語（トークン）に分ける
        #print(tokens_a)
        #tokens_a  =  [w for w in tokens_a if not w in stop_words]
        example_counter.append(len(tokens_a)  )#トークンの数のリスト

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]


        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        #print(tokens)
        segment_ids = [0] * len(tokens)



        input_ids = tokenizer.convert_tokens_to_ids(tokens)#idに変換
        #print(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length #max_seq_lengthと長さが異なっていたら止まる
        # pad sdp length here
        max_sdp_length  = 100
        position_mask =  [ int(e) + 1 for e in eval( sdp_postion ) ] #  "CLS " があるので+1する

        if len(position_mask) > max_sdp_length :
            position_mask = position_mask[:max_sdp_length  ]
        position_padding = [0] * (max_sdp_length - len(position_mask))

        position_mask +=position_padding#max_sdp_lengthまで配列を埋める(padding)
        #print(position_mask)
        #print(input_ids)
        assert len(position_mask) == max_sdp_length


        f_dict = {
            "input_ids" :input_ids,#文の単語をidに変換したもの
            "input_mask":input_mask,#マスク用(1)
            "type_ids" : segment_ids,#(0)
            "position_mask" : position_mask#マスクの場所
        }
        #print(position_mask)
        #features.append( input_ids,input_mask,segment_ids )
        features.append(f_dict )


    #print(features[0])
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
