import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from sklearn.model_selection import train_test_split

df = pd.read_csv("sentence_autofill_dataset.csv")
sentences = df['sentence'].tolist()
completions = df['completion'].tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = []
attention_masks = []
labels = []

for sentence, completion in zip(sentences, completions):

    masked_sentence = sentence.replace("[MASK]", tokenizer.mask_token)

    encoded = tokenizer.encode_plus(masked_sentence, 
                                     add_special_tokens=True, 
                                     max_length=50, 
                                     padding='max_length', 
                                     truncation=True, 
                                     return_tensors='pt')
    
    input_ids.append(encoded['input_ids'])
    attention_masks.append(encoded['attention_mask'])

    completion_encoded = tokenizer.encode(completion, 
                                          add_special_tokens=False, 
                                          max_length=50, 
                                          truncation=True)

    label_tensor = [tokenizer.pad_token_id] * 50
    for i in range(len(completion_encoded)):
        label_tensor[i] = completion_encoded[i]
    
    labels.append(torch.tensor(label_tensor).unsqueeze(0)) 

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.cat(labels, dim=0)

class SentenceDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_masks[idx], self.labels[idx]

train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
    input_ids, attention_masks, labels, test_size=0.1
)

train_dataset = SentenceDataset(train_inputs, train_masks, train_labels)
val_dataset = SentenceDataset(val_inputs, val_masks, val_labels)

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.train()

optimizer = AdamW(model.parameters(), lr=2e-5)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for epoch in range(3):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        batch_input_ids, batch_attention_masks, batch_labels = batch
        model.zero_grad()

        outputs = model(batch_input_ids, 
                        attention_mask=batch_attention_masks, 
                        labels=batch_labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

model.save_pretrained('./bert_next_word_completion_model')
tokenizer.save_pretrained('./bert_next_word_completion_model')
