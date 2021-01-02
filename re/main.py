
from pytorch_pretrained_bert import BertModel, BertTokenizer
import torch
import json
import logging

import sys
sys.path.append("./question-answering-from-transformers")

logging.getLogger().setLevel(logging.WARNING)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import  os
print(os.getcwd())
print("hello")

# the sentence in our case.
text = "rare bird has more than enough charm to make it memorable."

# get the tokenized words.
tokenizer = BertTokenizer.from_pretrained("/home/fwx/model_pytorch/bert_base_uncased")
words = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]

# load bert model
model = BertModel.from_pretrained("/home/fwx/model_pytorch/bert_base_uncased").to(device)
for param in model.parameters():
    param.requires_grad = False
model.eval()
print("hi")
# get the x (here we get x by hacking the code in the bert package)
tokenized_ids = tokenizer.convert_tokens_to_ids(words)
segment_ids = [0 for _ in range(len(words))]
token_tensor = torch.tensor([tokenized_ids], device=device)
segment_tensor = torch.tensor([segment_ids], device=device)
x = model.embeddings(token_tensor, segment_tensor)[0]

# here, we load the regularization we already calculated for simplicity
regularization = json.load(open("./question-answering-from-transformers/regular.json", "r"))

# extract the Phi we need to explain
def Phi(x):
    global model
    x = x.unsqueeze(0)
    attention_mask = torch.ones(x.shape[:2]).to(x.device)
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    # extract the 3rd layer
    model_list = model.encoder.layer[:3]
    hidden_states = x
    for layer_module in model_list:
        hidden_states = layer_module(hidden_states, extended_attention_mask)
    return hidden_states[0]

from Interpreter import Interpreter

interpreter = Interpreter(x=x, Phi=Phi, regularization=regularization).to(
    device
)


interpreter.optimize(iteration=5000, lr=0.01, show_progress=True)
interpreter.get_sigma()
# interpreter.visualize()