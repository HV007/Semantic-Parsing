import torch
import numpy as np
import json
from tqdm import tqdm
import sys
from transformers import BartForConditionalGeneration, BartTokenizerFast as BartTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TEST_FILE_PATH = sys.argv[1]
OUT_FILE_PATH = sys.argv[2]

print('Loading Best Model')
model = BartForConditionalGeneration.from_pretrained('./cs1190356_model/').to(device)
tokenizer = BartTokenizer.from_pretrained('./cs1190356_tokenizer/')

print('Loading Test Data')
val_data = []
with open(TEST_FILE_PATH, 'r') as f:
    for line in f:
        val_data.append(json.loads(line))

def generate_input(data):
    input_data = []
    for d in data:
        if 'output' in d.keys():
            del d['output']
        if 'history' in d.keys():
            del d['history']
        input_data.append(d['input'] + ' ' + json.dumps(d))
    return input_data

print('Generating Test Encoding')
val_encodings = tokenizer.batch_encode_plus(generate_input(val_data), truncation=True, max_length=1024, padding=True, return_tensors='pt')

# Create the dataset
class SemanticParsingDatasetTest(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return self.encodings['input_ids'][idx]

    def __len__(self):
        return len(self.encodings['input_ids'])

val_dataset = SemanticParsingDatasetTest(val_encodings)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

outf = open(OUT_FILE_PATH, 'w')

print('Generating Output')
with torch.no_grad():
    for batch in tqdm(val_loader):
        input_ids = batch.to(device)
        generated_ids = model.generate(input_ids, num_beams = 11, max_length = 250)
        generated_sentences = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for sent in generated_sentences:
            outf.write(sent + '\n')

outf.close()