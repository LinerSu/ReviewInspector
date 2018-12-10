
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import normalization
from normalization import text_normalization
from rnn_model import RNN
from biLSTM_model import BiLSTM
from cnn_model import CNN

parser = argparse.ArgumentParser(description='PyTorch Yelp ReviewInspector Model')

# Model parameters.
parser.add_argument('--data', type=str, default='../data/input',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='RNN', metavar='MD',
                    help='model: RNN, BILSTM, CNN')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111, metavar='S',
                    help='random seed')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size of training input')
parser.add_argument('--print_every', type=int, default=50,
                    help='every n iterations loss is printed')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--optimizer', type=str, default='SGD',
                    help='type of optimizer (SGD | Adam)')

args = parser.parse_args()
# parser.print_help()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

data, iter, INPUT_DIM, device = text_normalization(args.seed, args.batch_size)
train_data, val_data, test_data = data
train_iter, val_iter, test_iter = iter

def selectModel(): # Model
    if args.model == "RNN":
        model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
    elif args.model =="LSTM":
        model = BiLSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    return model

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division
    acc = correct.sum()/len(correct)
    return acc

def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for i, batch in enumerate(iterator):

        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

        if i % args.print_every == 0:
            print('[%d/%d] Loss: %.3f Acc: %.2f' % (i, len(iterator), loss.item(), acc.item()))

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():

        for batch in iterator:

            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def run(model):
    criterion = nn.BCEWithLogitsLoss()
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = model.to(device)
    criterion = criterion.to(device)

    print(model)

    N_EPOCHS = 5

    for epoch in range(N_EPOCHS):
        train_loss, train_acc = train(model, train_iter, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, val_iter, criterion)
        print(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% |')

    test_loss, test_acc = evaluate(model, test_iter, criterion)

    print("\n")

    print(f'| Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}% |')

def main():
    run(selectModel())

if __name__ == "__main__":
    main()
