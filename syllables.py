import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm

char_to_idx = {' ': 0, '\'': 1, '-': 2, '.': 3, 'a': 4, 'b': 5, 'c': 6, 'd': 7, 'e': 8, 'f': 9, 'g': 10, 'h': 11, 'i': 12, 'j': 13, 'k': 14, 'l': 15, 'm': 16, 'n': 17, 'o': 18, 'p': 19, 'q': 20, 'r': 21, 's': 22, 't': 23, 'u': 24, 'v': 25, 'w': 26, 'x': 27, 'y': 28, 'z': 29}
idx_to_char = {idx: char for char, idx in char_to_idx.items()} # TODO: needed?

def str_to_tensor(word):
    return torch.LongTensor([char_to_idx[letter] for letter in word])

class SyllableCounter(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = self.embed(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def collate_fn(batch):
    sequences = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=char_to_idx[' '])
    labels = torch.LongTensor([item[1] for item in batch])

    return sequences, labels

class SyllableDataset(Dataset):
    def __init__(self, data, labels):
        super().__init__()

        self.data = [str_to_tensor(word) for word in data]
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def main():
    with open('data/phoneticDictionary.csv') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        data = []
        labels = []
        for row in csv_reader:
            data.append(row[1])
            labels.append(int(row[3]))
    

    train_data, val_data, test_data = random_split(SyllableDataset(data, labels), [0.70, 0.15, 0.15])
    train_loader = DataLoader(train_data, 32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, 32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, 32, shuffle=False, collate_fn=collate_fn)

    vocab_size = 30 # According to tests
    embedding_dim = 50
    hidden_dim = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SyllableCounter(vocab_size, embedding_dim, hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()

        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1} / {num_epochs}'):
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.float())
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for inputs, targets in val_loader:

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), targets.float())
                val_loss += loss.item()

                predicted = torch.round(outputs.squeeze()).long()
                correct_preds += (predicted == targets).sum().item()
                total_preds += targets.size(0)

        
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {(correct_preds / total_preds):.2f}")
    
    model.eval()
    test_loss = 0.0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for inputs, targets in test_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets.float())
            test_loss += loss.item()

            predicted = torch.round(outputs.squeeze()).long()
            correct_preds += (predicted == targets).sum().item()
            total_preds += targets.size(0)
    
    print(f"Test Loss: {test_loss / len(test_loader)}, Test Accuracy: {(correct_preds / total_preds):.2f}")


if __name__ == '__main__':
    main()