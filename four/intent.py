import torch
from torch import nn
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW
import json
import numpy as np
import random
#定义模型
model_config="hfl/chinese-bert-wwm-ext"
device="cpu"
intent_vocab='intent_vocal.json'
class intentbert(nn.Module):
    def __init__(self,device,intent_dim,intent_weight=None):
        super(intentbert,self).__init__()
        self.label=intent_dim
        self.intent_weight = (
            intent_weight
            if intent_weight is not None
            else torch.tensor([1.0] * intent_dim)
        )
        self.bert=BertModel.from_pretrained(model_config)
        self.dropout=nn.Dropout(0.3)
        self.hidden_units=768
        self.intent_hidden=nn.Linear(self.bert.config.hidden_size,self.hidden_units)
        self.classifier=nn.Linear(self.hidden_units,self.label)
        self.loss=torch.nn.BCEWithLogitsLoss(pos_weight=self.intent_weight)

    def forward(self,seq_tensor,mask_tensor,input_tensor):
        bert_outputs=self.bert(input_ids=seq_tensor, attention_mask=mask_tensor)
        output=bert_outputs[1]
        output=nn.functional.relu(self.intent_hidden(self.dropout(output)))
        logit=self.classifier(output)
        loss=self.loss(logit,input_tensor)
        return logit,loss

tokenizer = BertTokenizer.from_pretrained(model_config)
def bert_tokenize(word_seq):
    split_tokens = []
    basic_tokens = tokenizer.basic_tokenizer.tokenize(" ".join(word_seq))
    accum = ""
    i, j = 0, 0
    for i, token in enumerate(basic_tokens):
        if (accum + token).lower() == word_seq[j].lower():
            accum = ""
        else:
            accum += token
        for sub_token in tokenizer.wordpiece_tokenizer.tokenize(
                basic_tokens[i]
        ):
            split_tokens.append(sub_token)
        if accum == "":
            j += 1
    return split_tokens

vocab_datas=json.load(open(intent_vocab,encoding="utf-8"))
intent_dim = len(vocab_datas)
intent2id = dict([(x, i) for i, x in enumerate(vocab_datas)])
intent_weight = [1] * len(intent2id)
def seq_intent2id(intents):
    return [intent2id[x] for x in intents if x in intent2id]


#加载数据
# max_len=128
leng=60
datas=json.load(open("intent_train_data.json",encoding="utf-8"))
sen_len=[]
#数据截取
train_size=len(datas)
for data in datas:
    max_sen_len = max(max_sen_len, len(data[0]))
    sen_len.append(len(data[0]))
    # 对每条数据进行长度的截取
    data[0] = data[0][:leng]
    #token处理
    word_seq = bert_tokenize(data[0])
    data.append(word_seq)
    #转换id
    data.append(seq_intent2id(data[1]))
    for intent_id in data[-1]:
        intent_weight[intent_id]+=1
for intent, intent_id in intent2id.items():
    neg_pos = (train_size - intent_weight[intent_id]) / intent_weight[intent_id]
    intent_weight[intent_id] = np.log10(neg_pos)
    intent_weight = torch.tensor(intent_weight)

#
def pad_batch(batch_data):
    batch_size = len(batch_data)
    max_seq_len = max([len(x[0]) for x in batch_data]) + 2
    word_mask_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)

    intent_tensor = torch.zeros((batch_size, intent_dim), dtype=torch.float)  #

    for i in range(batch_size):
        words = batch_data[i][0]
        intents = batch_data[i][-1]
        words = ["[CLS]"] + words + ["[SEP]"]
        indexed_tokens = tokenizer.convert_tokens_to_ids(words)
        sen_len = len(words)
        word_seq_tensor[i, :sen_len] = torch.LongTensor([indexed_tokens])
        word_mask_tensor[i, :sen_len] = torch.LongTensor([1] * sen_len)

        for j in intents:
            intent_tensor[i, j] = 1.0
    return word_seq_tensor, word_mask_tensor, intent_tensor


def get_train_batch(batch_size):
    batch_data = random.choices(datas, k=batch_size)
    return pad_batch(batch_data)

#model
model=intentbert(device,intent_dim,intent_weight)
model.to(device)
#optimizer
optimizer=AdamW(lr=0.01,eps=1e-8)
max_step=500
batch_size=50
train_loss=0
for step in range(1,max_step+1):
    model.train()
    batched_data=get_train_batch(batch_size)
    word_seq_tensor, word_mask_tensor, intent_tensor = batched_data
    intent_logits, intent_loss = model.forward(
        word_seq_tensor, word_mask_tensor, intent_tensor
    )

    train_intent_loss += intent_loss.item()
    loss = intent_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.zero_grad()
    if step % 100 == 0:
        train_intent_loss = train_intent_loss / 100
        print("[%d|%d] step" % (step, max_step))
        print("\t intent loss:", train_intent_loss)




