import csv
import argparse as ap

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
    def __init__(self, vocab_size, embedding_dim, hidden_dim, lstm_layers, dropout):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=lstm_layers, dropout=dropout, batch_first=True)
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


def parse_args():
    parser = ap.ArgumentParser(prog='python3 syllables.py', description="Train or use an AI built to recognize the amount of syllables in a given word.")

    mutex_train = parser.add_mutually_exclusive_group(required=True)
    mutex_train.add_argument('-t', '--train', action='store_true', help='Train a new model if specified.')
    mutex_train.add_argument('-e', '--eval', '--evalutate', type=str, help='Evaluate given word on model.')
    # TODO: maybe make it so that a model can be loaded and saved in the same program call, something like update_model
    mutex_path = parser.add_mutually_exclusive_group()
    mutex_path.add_argument('-s', '--save', '--save-path', type=str, default='data/syllable_model.pth', help='Path to file where model state_dict should be stored.')
    mutex_path.add_argument('-l', '--load', '--load-path', type=str, default='data/phoneticDictionary.csv', help='Path to dataset.')

    hyperparams = parser.add_argument_group('Hyper parameters')
    hyperparams.add_argument('-n', '--epochs', '--num-epochs', type=int, default=5, help='Number of epochs the model will be trained for.')
    hyperparams.add_argument('-b', '--batch', '--batch-size', type=int, default=32, help='Size of a batch processed during training.')
    hyperparams.add_argument('--embeds', '--embed-dims', '--embedding-dimensions', type=int, default=50, help='Size of embedding dimensions in neural network.')
    hyperparams.add_argument('--hiddens', '--hidden-layers', type=int, default=128, help='Amount of hidden layers in LSTM.')
    hyperparams.add_argument('--layers', '--lstm-layers', type=int, default=2, help='Amount of layers in the networks LSTM.')
    hyperparams.add_argument('--lr', '--learning-rate', type=float, default=0.001, help='Learning rate for optimizer Adam.')
    hyperparams.add_argument('-d', '--dropout', type=float, default=0.3, help='Dropout rate during learning.')

    return parser.parse_args()


def main():
    args = parse_args()



    with open(args.load) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        data = []
        labels = []
        for row in csv_reader:
            data.append(row[1])
            labels.append(int(row[3]))
    

    train_data, val_data, test_data = random_split(SyllableDataset(data, labels), [0.70, 0.15, 0.15])
    train_loader = DataLoader(train_data, args.batch, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, args.batch, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, args.batch, shuffle=False, collate_fn=collate_fn)

    vocab_size = 30 # According to tests, dependant on used dataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SyllableCounter(vocab_size, args.embeds, args.hiddens, args.layers, args.dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()

        for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch + 1} / {args.epochs}'):
            
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