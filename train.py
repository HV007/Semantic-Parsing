import torch
import numpy as np
import json
from tqdm import tqdm
import sys
from transformers import BartForConditionalGeneration, BartTokenizerFast as BartTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAIN_FILE_PATH = sys.argv[1]
DEV_FILE_PATH = sys.argv[2]

print('Loading Model')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

print('Loading Data')
train_data = []
with open(TRAIN_FILE_PATH, 'r') as f:
    for line in f:
        train_data.append(json.loads(line))

val_data = []
with open(DEV_FILE_PATH, 'r') as f:
    for line in f:
        val_data.append(json.loads(line))

# Tokenize the data

def generate_input(data):
    input_data = []
    for d in data:
        if 'output' in d.keys():
            del d['output']
        if 'history' in d.keys():
            del d['history']
        input_data.append(d['input'] + ' ' + json.dumps(d))
    return input_data

print('Generating Encoding')
train_labels = tokenizer.batch_encode_plus([d['output'] for d in train_data], truncation=True, max_length=1024, padding=True, return_tensors='pt')
train_encodings = tokenizer.batch_encode_plus(generate_input(train_data), truncation=True, max_length=1024, padding=True, return_tensors='pt')
val_labels = tokenizer.batch_encode_plus([d['output'] for d in val_data], truncation=True, max_length=1024, padding=True, return_tensors='pt')
val_encodings = tokenizer.batch_encode_plus(generate_input(val_data), truncation=True, max_length=1024, padding=True, return_tensors='pt')

# Create the dataset
class SemanticParsingDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return (self.encodings['input_ids'][idx], self.labels['input_ids'][idx])

    def __len__(self):
        return len(self.labels['input_ids'])

train_dataset = SemanticParsingDataset(train_encodings, train_labels)
val_dataset = SemanticParsingDataset(val_encodings, val_labels)

# Create the dataloader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

lr = 1e-5
num_epochs = 15

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_loss = 1e9

print('Training')

for epoch in range(num_epochs):
    print('EPOCH:', epoch)
    running_loss = 0.0
    model.train()
    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch[0].to(device)
        labels = batch[1].to(device)
        outputs = model(input_ids, labels=labels)
        loss = outputs[0]
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    print('Train Loss:', running_loss / len(train_loader))
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_loader):
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(input_ids, labels=labels)
            loss = outputs[0]
            running_loss += loss.item()
    print('Val Loss:', running_loss / len(val_loader))
    if running_loss / len(val_loader) < best_val_loss:
        best_val_loss = running_loss / len(val_loader)
        tokenizer.save_pretrained('./cs1190356_tokenizer/')
        model.save_pretrained('./cs1190356_model/')
        print('Model Saved!')
