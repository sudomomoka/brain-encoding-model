import glob
import re
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import unicodedata
import string
import sys
import codecs
import random
import collections
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import itertools
import pickle
import scipy.stats
from io import open
from sklearn.metrics import f1_score
from sklearn import datasets, linear_model

from gensim.models import KeyedVectors
from transformers import BertTokenizer, BertModel
from transformers import BertConfig
import matplotlib
import MeCab
#import unidic
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, GroupKFold, GroupShuffleSplit, KFold
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from optimizer_Adamw import AdamW, LambdaLR, get_linear_schedule_with_warmup

print(torch.nn.utils.clip_grad_norm_)
print("cudnn version", torch.backends.cudnn.version())
device = "cuda" if torch.cuda.is_available() else 'cpu'
#device = torch.device("cuda", index=1 if torch.cuda.is_available() else "cpu")

print(device)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())



#bertモデルの読み込み
#config = BertConfig.from_pretrained('../timebank_model/bert-base-uncased-jap/',output_hidden_states=True)
#print(config)
#model = BertModel.from_pretrained('../timebank_model/bert-base-uncased-jap/', output_hidden_states=True)
#state_dict  = torch.load("../timebank_model/timebank.pt")["state"]
#model = model.to(device)
tokenizer = BertTokenizer.from_pretrained('../../TemporalRelation/pytorch_model/')


#正規化
def Standardization(X, mean, sd):

	return (float(X - mean)/sd)

def unicodeToUtf8(s):
    return s.encode('utf-8')

def readlines(filename):
    f = codecs.open(filename, 'r', 'utf-8')
    lines = f.read().strip().split('\n')
    #print(lines)
    f.close()
    return [unicodeToUtf8(line) for line in lines]

def read_multi_files(files_namelist):
    files_list = []
    for filename in files_namelist:
        # print(filename)
        each_file_line_list = readlines(filename)
        files_list += each_file_line_list
    # return [unicodeToUtf8(line) for line in lines if 'reltype' in line]
    return files_list

def create_data_loader(doc_tensor_dict, batch_size=16):
    train_dataset = doc_tensor_dict["train_dataset"]
    test_dataset = doc_tensor_dict["test_dataset"]
    # train_dataset, valid_dataset = split_hanshu(  train_dataset  )
    train_split = 0.85
    train_size = int(train_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    return train_loader, valid_loader, test_loader

def doc_kfold(data_dir, cv=5):
    file_list, file_splits = [], []
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".txt"):
            dir_file = os.path.join(data_dir, file)
            file_list.append(dir_file)
    #    logger.info("[Number] %i files in '%s'" % (len(file_list), data_dir))
    gss = KFold(n_splits=cv, shuffle=True, random_state=1029)
    for train_split, test_split in gss.split(file_list):
        file_splits.append((
            [file_list[fid] for fid in train_split],
            [file_list[fid] for fid in test_split]
        ))
    return file_splits


def convert_full_sequence_cls_to_features(examples):

    #print(model)
    sen = []
    sent = ""
    cls_counter = []
    tokens = []
    bert_tokens = []
    l = ""
    no =0

    sec=0
    counter = 0
    number_of_sentence = 0
    #width = 1#BERTのとる時間幅

    tagger = MeCab.Tagger()

    rm=[0]

    #print(rm)
    print("文章の数",len(examples))

    counter =0
    c=0

    for l in examples:
        if l != '..':
            sen = l
            node = tagger.parseToNode(sen)
            sent = "[CLS]" + sent
            while node:
                if len(node.feature.split(",")) > 6:
                        #print(sen)
                        #print(node.feature.split(","))
                        if node.feature.split(",")[6] != '*':
                                sent = sent + " " + node.feature.split(",")[6]
                        node = node.next
                else:
                        node = node.next
            sent = sent + " " + "[SEP]"
            token = sent.split( )
            tokens.append(token)
            sent = ""
        else:
            sent = "[CLS]" + " " + "0" +  " " + "[SEP]"
            token = sent.split( )
            tokens.append(token)
            sent = ""

    max_seq_length = 100 #BERTに入れることのできる最大のtoken数
    couter = 0

    #if len(tokens) > max_seq_length:
    #    tokens = tokens[:(max_seq_length)]
    #print(len(tokens))

    #print(list(itertools.chain.from_iterable(bert_tokens)))
    #print(sum(tokens, []))
    bert_tokens = sum(tokens, [])

    #print(bert_tokens)
    #print("token length: ",len(bert_tokens))
    print("sentenceの数:",len(tokens))


    input_ids = []
    input_ids2 = []
    counter = 0
    a = 0
    flag = 0
    #print("token数を512で割った時の商: ",len(bert_tokens) // 512 ,"\ntoken数を512で割った時のあまり: ",len(bert_tokens) % 512)

    """
    while flag < len(bert_tokens):
        if counter < 512:
            input_ids.append(bert_tokens[flag])
            counter += 1
            flag += 1
        else :
            while input_ids[-1] != '[SEP]':
                input_ids = input_ids[:-1]
                a += 1
            flag = flag-a
            input_ids2.append(input_ids)
            #print(len(input_ids))
            input_ids = []
            counter = 0
            a = 0
    input_ids2.append(input_ids)
    #print(len(input_ids))
    input_ids =[]

    print(input_ids2)
    """

    for i in bert_tokens:
        if i != '[SEP]':
            input_ids.append(i)
        else:
            input_ids.append(i)
            input_ids2.append(input_ids)
            #print(input_ids)
            input_ids = []
    print(len(input_ids2))

    couter = 0
    features = []

    for i in range(len(input_ids2)):
        pre_input_ids=tokenizer.convert_tokens_to_ids(input_ids2[i])#idに変換
        #print(pre_input_ids)
        padding = [0] * (max_seq_length - len(pre_input_ids))
        pre_input_ids += padding
        #print(pre_input_ids.shape)
        #np.append(input_ids,pre_input_ids,axis=0)
        input_ids.append(pre_input_ids)


    #print(len(input_ids))
    input_ids = np.array(input_ids)

    print("input idのサイズ: ",input_ids.shape)
    print("input id:\n",input_ids)
    #attention_mask = np.where(input_ids != 0,1,0)
    #print("attention maskのサイズ: ",attention_mask.shape)
    #print("attention mask:\n",attention_mask)

    return input_ids

    #cls_flag = np.where(input_ids == 2, 1,0)
    #print("cls_flag: ",cls_flag)
    #print("cls_flag: ",cls_flag.shape)

    #input_ids = torch.tensor(np.array(input_ids))
    #attention_mask = torch.tensor(attention_mask)
    #print(tokens_tensor)
    #print(input_ids.shape)

    """
    counter = 0
    flag = 0
    fat = []
    features = []

    for i in range(len(input_ids)):
        for k in range(max_seq_length):
            if cls_flag[i][k] == 1:
                #counter += 1
                #print("[CLS]position:",i,k)
                fat.append(bert_outputs[2][-2][i,k,:].numpy())

    #print(counter)
    fat = np.array(fat)
    #features = bert_outputs[0][:,counter,:].numpy()
    #print(features)


    #fat = scipy.stats.zscore(fat,axis=None)#正規化

    #標準化
    mean = np.mean(fat)
    sd = np.std(fat)
    fat = np.array([[Standardization(X, mean, sd) for X in row] for row in fat])

    print("fat shape:",fat.shape)
    #print("fat type:",type(f))

    counter = 0
    c=0
    a=0


    #文がない部分は０のnumpy配列を入れる
    list = [0]*768
    for i in cls_counter:
        if i[0] == 'yes':
            #a = a+i[1]
            c = int(i[1])
            while c>0:
                features.append(fat[counter])
                c -= 1
            counter += 1
        if i[0] == 'no':
            #print(i[1])
            a = a+int(i[1])
            c = int(i[1])
            while c>0:
                features.append(list)
                c -= 1

    print(a)
    features = np.array(features)
    print(features)
    #print(features[2])
    #print(features[7])
    #print(features[8])
    print("features shape:",features.shape)
    """

    #with open("../data/pickle_data/DreamGirls_sentence.pickle",'wb') as f:
    #    pickle.dump(features,f)



def getSentAndCategory(datalist):
    global sent_list
    global category_list
    sent_list = []

    category_list = []
    for data in datalist:
        # sent_list.append((data[3],data[5]))
        #print(data)
        #print(data[3])
        sent_list.append(data[2])
        category_list.append(data[3])

    n_categories = len(set(category_list))

    #print("category_list",category_list)
    #print(n_categories)
    return (sent_list, category_list, n_categories)

def get_sequence_2_TensorDataset(cat2ix, x_train, y_train, x_test, y_test):
    x_train_token_ids = []
    x_train_type_ids = []
    x_train_mask_ids = []

    x_test_token_ids = []
    x_test_type_ids = []
    x_test_mask_ids = []
    max_seq_length = 100
    #from bert_process_functions import convert_tokens_to_features, convert_full_sequence_sdp_to_features
    # train_features  =  convert_tokens_to_features ( x_train   , max_seq_length, tokenizer )
    #print(x_train)

    train_features = convert_full_sequence_cls_to_features(x_train)
    test_features = convert_full_sequence_cls_to_features(x_test)
    print(len(train_features))
    print(len(test_features))

    tr_input_ids = torch.tensor(train_features)
    #tr_input_mask = torch.tensor([d["input_mask"] for d in train_features])
    #tr_type_ids = torch.tensor([d["type_ids"] for d in train_features])
    #tr_position_mask = torch.tensor([d["position_mask"] for d in train_features])

    te_input_ids = torch.tensor(test_features)
    #te_input_mask = torch.tensor([d["input_mask"] for d in test_features])
    #te_type_ids = torch.tensor([d["type_ids"] for d in test_features])
    #te_position_mask = torch.tensor([d["position_mask"] for d in test_features])

    print(tr_input_ids)
    print(te_input_ids)

    y_train_t = torch.tensor([cat2ix[y] for y in y_train])
    y_test_t = torch.tensor([cat2ix[y] for y in y_test])
    # print("y_test_t",y_test_t)
    train_data = TensorDataset(tr_input_ids, y_train_t)
    test_data = TensorDataset(te_input_ids, y_test_t)

    doc_tensor_dict = {
        "train_dataset": train_data,
        "test_dataset": test_data
    }

    return doc_tensor_dict


def process_data(train_path_list, test_path_list):
    datalist_train = [re.split('\t|\|', line.decode('utf-8')) for line in read_multi_files(train_path_list)]
    datalist_test = [re.split('\t|\|', line.decode('utf-8')) for line in read_multi_files(test_path_list)]
    #print(datalist_train)

    sent_list_train, category_list_train, n_train_categories = getSentAndCategory(datalist_train)
    sent_list_test, category_list_test, n_test_categories = getSentAndCategory(datalist_test)
    sent_list_all, category_list, _ = getSentAndCategory(datalist_train + datalist_test)
    sent2category = {}


    for i in range(len(sent_list)):
        sent2category[sent_list[i]] = category_list[i]
    # print("sent:", sent_list_test[0][0].encode('utf-8').strip())
    # print(np.shape(sent_list_test[0]))
    x_train = np.array(sent_list_train)
    y_train = np.array(category_list_train)

    x_test = np.array(sent_list_test)
    y_test = np.array(category_list_test)



    cat2ix = get_cat2ix()
    doc_tensor_dict = get_sequence_2_TensorDataset(cat2ix, x_train, y_train, x_test, y_test)

    print(doc_tensor_dict)

    return cat2ix, doc_tensor_dict


#時制にラベルをつける
def get_cat2ix():
    cat2ix = {}
    c = collections.Counter(category_list)
    print(c.most_common())
    for cat in category_list:
        if cat not in cat2ix :
            cat2ix[cat] = len(cat2ix)
    print(cat2ix)
    return cat2ix

def InitialModel(n_epoch, len_cat2ix, train_loader, concat_all=True, with_lstm=True):
    model = define_model(len_cat2ix, concat_all=concat_all, with_lstm=with_lstm)
    # print("model parameters : ")
    max_grad_norm = 1.0
    num_training_steps = n_epoch * len(train_loader)
    num_warmup_steps = num_training_steps / 10

    param_optimizer = list(model.named_parameters())
    bert_name_list = ['bert_model']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in bert_name_list)], 'lr': 2e-5},
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in bert_name_list)], 'lr': 2e-5}
    ]

    # To reproduce BertAdam specific behavior set correct_bias=False
    optimizer = AdamW(
        optimizer_grouped_parameters,
        correct_bias=False
    )

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    return model, optimizer, criterion, scheduler


class SaveOutput:
  def init(self):
     self.outputs = []
  def __call__(self, module, module_in, module_out):
    self.outputs.append(module_out.detach())
  def clear(self):
    self.outputs = []




class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, concat_all=True, with_lstm=True):
        super(BiLSTM, self).__init__()

        self.save_output = SaveOutput()
        hook_handles = []
        #layer = self.model.bert_model.encoder.layer.11.output.LayerNorm.bias[0]     # whatever you want to snoop
        #handle = layer.register_forward_hook(self.save_output)
        #hook_handles.append(handle)

        self.hidden_size = hidden_size
        self.output_size = output_size
        print("hidden size: ",hidden_size)
        print("input_size : ", input_size)
        self.concat_all = concat_all
        self.with_lstm = with_lstm
        if self.concat_all:
            self.input_size = 4 * input_size
        else:
            self.input_size = 1 * input_size
        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=True,
                            dropout=0.3)
        if self.with_lstm:
            self.hidden = nn.Linear(hidden_size * 2, hidden_size * 2)
            self.out = nn.Linear(hidden_size * 2, output_size)

        else:
            if self.concat_all:
                self.hidden = nn.Linear(768 * 4, hidden_size)
                self.out = nn.Linear(hidden_size * 1, output_size)

            else:
                self.hidden = nn.Linear(768,hidden_size)
                self.out = nn.Linear(hidden_size * 1, output_size)

        self.bert_model = BertModel.from_pretrained('../../TemporalRelation/pytorch_model/', output_hidden_states=True)
        #self.bert_model = BertModel.from_pretrained('../東北大bertmlm', output_hidden_states=True)
        #self.bert_model = BertModel.from_pretrained('../timebank_model/bert-base-uncased-jap/', output_hidden_states=True)
        #state_dict  = torch.load("../../日本語話し言葉/model/話し言葉_funabiki_model.pt")
        #self.bert_model.load_state_dict(state_dict, strict=False)
        #bert_model = bert_model.to(device)
        self.freeze(False)

    def freeze(self, flag=False):
        if flag is True:
            for name, param in self.bert_model.named_parameters():
                if 'classifier' not in name:  # classifier layer
                    param.requires_grad = False
        else:
            for name, param in self.bert_model.named_parameters():
                param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, position_mask=None, token_type_ids=None, hidden_state=None):
        # with torch.no_grad():
        output_bert = self.bert_model(input_ids)
        #print("output_bert 0 :",output_bert[0].shape)
        #print("output_bert 1 :",output_bert[1].shape)
        #print("output_bert 2 :",len(output_bert[2]))
        #print("output_bert 2 -1 :",output_bert[2][-1].shape)
        #exit()

        if self.concat_all:
            bert_emebddings = torch.cat(tuple([output_bert[2][i] for i in [-1, -2, -3, -4]]), dim=-1)

        else:
            bert_emebddings = output_bert[2][-1][0:]
            #print(bert_emebddings.shape)
            #exit()

        #print("position_mask:",position_mask[1])
        #sdp_bert_embeddings = batched_index_select(bert_emebddings, 1, position_mask)
        #embedding = torch.add(output_bert[2][-1][0:],sdp_bert_embeddings)
        #print(embedding.shape)
        #print(sdp_bert_embeddings.shape)
        #exit()
        #print(len(bert_emebddings))
        #for i in range(bert_emebddings)
        #embeddings = batched_index_select(bert_emebddings, 1, position_mask)[0]
        #print ( "embeddings shape ", embeddings.shape   )
        if self.with_lstm:
            output, hidden_state = self.lstm(sdp_bert_embeddings.to(device), hidden_state)
            output, _ = torch.max(F.relu(output), 1)
        else:
            #print(sdp_bert_embeddings.shape)
            output, _ = torch.max(F.relu(bert_emebddings), 1)
        #print ( "max out ", output.shape )
        hid = self.hidden(output)
        output = self.hidden(output)
        #print ( "hidden  out ", output.shape )
        #print(output)

        output = self.out(F.relu(output))
        #print ( "relu  out ", output.shape )
        #print(output)

        output = F.log_softmax(output, dim=-1)
        #print ( "log_softmax.shape ",  output.shape   )
        #print(output)

        #exit()
        #features = self.save_output.outputs[0].to('gpu').detach().numpy().copy()
        #self.save_output.clear()

        return output , hid

    def initHidden(self, batch_size):
        h = torch.zeros(2, batch_size, self.hidden_size).to(device)
        c = torch.zeros(2, batch_size, self.hidden_size).to(device)
        return (h, c)


def define_model(len_cat2ix, concat_all=True, with_lstm=True):
    n_hidden = 200
    n_epoch = 20
    model = BiLSTM(768, n_hidden, len_cat2ix, concat_all=concat_all, with_lstm=with_lstm)
    model = model.to(device)

    return model


def train_model(n_epoch=10, train_loader=None, valid_loader=None, test_loader=None,
                model=None, optimizer=None, criterion=None,
                scheduler=None, flag=None):
    loss_list = []
    acc_list = []
    for epoch in range(1, n_epoch + 1):

        epoch_loss = []
        model.train()
        y_pred = []
        y_true = []
        best_loss_1 = 100000
        best_loss_2 = 100000

        for batch_data in tqdm(train_loader):
            batch_data = (e.to(device) for e in batch_data)
            input_bert_ids,y = batch_data

            model.train()
            model.zero_grad()
            batch_size_ = input_bert_ids.shape[0]
            hidden_state = model.initHidden(batch_size_)

            # output = model(input_bert_ids.to(device), token_type_ids.to(device), hidden_state)
            output, _= model(input_bert_ids, hidden_state)
            #print(output.shape)
            y = y.to(device)
            loss = criterion(output, y)
            loss.backward()
            max_grad_norm = 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            pred = torch.argmax(output, dim=-1)

            y_true.append(y.detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())
            epoch_loss.append(loss.item())

        y_true = np.array(list(itertools.chain(*y_true)))
        y_pred = np.array(list(itertools.chain(*y_pred)))
        accuracy = accuracy_score(y_true, y_pred)
        print('epoch %i, loss=%.4f, acc=%.2f%%' % (
            epoch, sum(epoch_loss) / len(epoch_loss), accuracy * 100))

        model.eval()
        with torch.no_grad():

            epoch_loss_val = []
            y_pred_val = []
            y_true_val = []
            for batch_data in valid_loader:
                batch_data = (e.to(device) for e in batch_data)
                input_bert_ids, y = batch_data

                batch_size_ = input_bert_ids.shape[0]
                hidden_state = model.initHidden(batch_size_)

                # output = model(input_bert_ids.to(device), token_type_ids.to(device), hidden_state)
                output, _ = model(input_bert_ids,hidden_state)
                #print(output.shape)
                y = y.to(device)
                loss_val = criterion(output, y)
                pred = torch.argmax(output, dim=-1)
                y_true_val.append(y.detach().cpu().numpy())
                y_pred_val.append(pred.detach().cpu().numpy())
                epoch_loss_val.append(loss.item())

            y_true_val = np.array(list(itertools.chain(*y_true_val)))
            y_pred_val = np.array(list(itertools.chain(*y_pred_val)))
            acc_val = accuracy_score(y_true_val, y_pred_val)
            print('epoch %i, loss=%.4f, acc=%.2f%%' % (
                epoch, sum(epoch_loss_val) / len(epoch_loss_val), acc_val * 100))
            loss_list.append(sum(epoch_loss_val) / len(epoch_loss_val))
            acc_list.append(acc_val)

            #########
            # global best_loss_1, best_loss_2

            if loss_val < best_loss_1 and flag == 1:
                torch.save(model.state_dict(), '../model/model1')
                best_loss_1 = loss_val

            if loss_val < best_loss_2 and flag == 2:
                torch.save(model.state_dict(), '../model/mentalist_model')
                best_loss_2 = loss_val

        model.eval()
        with torch.no_grad():
            y_pred = []
            y_true = []
            feature = []

            for batch_data in test_loader:
                batch_data = (e.to(device) for e in batch_data)
                input_bert_ids, y = batch_data
                batch_size_ = input_bert_ids.shape[0]
                hidden_state = model.initHidden(batch_size_)

                output , _ = model(input_bert_ids,hidden_state)
                #print(output)
                #print(output.shape)
                y = y.to(device)
                pred = torch.argmax(output, dim=-1)
                #print(pred)
                y_true.append(y.detach().cpu().numpy())
                y_pred.append(pred.detach().cpu().numpy())
                #feature.append(hid.detach().cpu().numpy())
            y_true = np.array(list(itertools.chain(*y_true)))
            y_pred = np.array(list(itertools.chain(*y_pred)))
            #feature = np.array(list(itertools.chain(*feature)))
            #print(y_pred)
            #print(feature.shape)
            acc = accuracy_score(y_true, y_pred)
            from sklearn.metrics import f1_score
            f1_macro = f1_score(y_true, y_pred, average='macro')
            f1_micro = f1_score(y_true, y_pred, average='micro')
            f1_weighted = f1_score(y_true, y_pred, average='weighted')
            results = {
                "acc": acc,
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "f1_weighted": f1_weighted,

            }
            print("------------------")
            print("test result", acc * 100)
            print("------------------")

    return loss_list, acc_list


def evaluate_model(model, test_loader):
    #print(model)
    model.eval()

    with torch.no_grad():
        y_pred = []
        y_true = []
        feature = []

        for batch_data in test_loader:
            # print(batch_data)
            batch_data = (e.to(device) for e in batch_data)
            input_bert_ids, y = batch_data
            batch_size_ = input_bert_ids.shape[0]
            hidden_state = model.initHidden(batch_size_)

            output ,hid= model(input_bert_ids,hidden_state)
            #print("hid:",hid)

            #print(output.shape)
            #print(output[1])
            #print(output[1].shape)
            y = y.to(device)
            pred = torch.argmax(output, dim=-1)

            y_true.append(y.detach().cpu().numpy())
            y_pred.append(pred.detach().cpu().numpy())
            # print("****************y_pred*********",np.shape(y_pred))
            feature.append(hid.detach().cpu().numpy())

        y_true = np.array(list(itertools.chain(*y_true)))
        y_pred = np.array(list(itertools.chain(*y_pred)))
        feature = np.array(list(itertools.chain(*feature)))

        acc = accuracy_score(y_true, y_pred)
        from sklearn.metrics import f1_score
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        results = {
            "acc": acc,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,

        }
        return results, acc ,feature

results_cv_list1 = list()
results_cv_list2 = list()
kfold_num=1

data_dir = 'dct_words'
data_splits = doc_kfold(data_dir, cv=5)
acc_all1 = []
acc_all2 = []
acc_features2 = []
for cv_index in range(kfold_num):
    if cv_index < -4:
        continue
    else:

        print("run the ")
        train_path = data_splits[cv_index][0]
        test_path = data_splits[cv_index][1]

        train_path_list = data_splits[cv_index][0]
        test_path_list = data_splits[cv_index][1]
        #print(len(train_path_list), train_path_list)
        #print(len(test_path_list), test_path_list)

        train_path_list = ['../data/Heroes/Heroes_run_id.txt', '../data/Glee/Glee_run_id.txt','../data/DreamGirls/DreamGirls_run_id.txt','../data/GIS1/GIS1_run_id.txt','../data/GIS2/GIS2_run_id.txt']
        test_path_list = ['../data/Mentalist/Mentalist_run_id.txt']

        print(len(train_path_list), train_path_list)
        print(len(test_path_list), test_path_list)

        cat2ix, doc_tensor_dict = process_data(train_path_list, test_path_list)
        train_loader, valid_loader, test_loader = create_data_loader(doc_tensor_dict, batch_size=16)
        print("length of train_loader:", len(train_loader))

        # define model and optimizer
        n_epoch = 20

        model2, optimizer2, criterion2, scheduler2 = InitialModel(n_epoch, len(cat2ix), train_loader,
                                                                  concat_all=False, with_lstm=False)

        loss_list2, acc_list2 = train_model(n_epoch, train_loader=train_loader, valid_loader=valid_loader,
                                            test_loader=test_loader,
                                             model=model2,
                                            optimizer=optimizer2,
                                            criterion=criterion2,
                                            scheduler=scheduler2, flag=2)
        model2.load_state_dict(torch.load('../model/mentalist_model'),strict=False)
        results2, acc_2 ,feature2= evaluate_model(model2, test_loader)
        print("feature shape:",feature2.shape)
        acc_all2.append(acc_2)
        acc_features2.append(feature2)
        results_cv_list2.append(results2)

df2 = pd.DataFrame(results_cv_list2)
print(test_path_list)
print(" results cv is : ")
print(df2)
#df1.to_csv('result_cv.tsv1', float_format='%.2f', sep='\t')
df2.to_csv('result_cv.tsv2', float_format='%.2f', sep='\t')
#print("acc_average yes:", sum(acc_all1) / kfold_num)
print("acc_average no:", sum(acc_all2) / kfold_num)
features = np.array(acc_features2).mean(axis=0)
print("features shape:", features.shape)


#with open("話し言葉_dvd_funabiki/DreamGirls_feature",'wb') as f:
with open("../data/only_dvd_data/Mentalist_feature",'wb') as f:
    pickle.dump(features,f)

