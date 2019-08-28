"""
    Load + visualize
"""
# Lib
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch
from collections import Counter
from string import punctuation
import numpy as np

# Read data
with open("data/reviews.txt", 'r') as f:
    reviews = f.read()
with open("data/labels.txt", 'r') as f:
    labels = f.read()

"""
    Data-preprocessing
"""

# get rid of punctuation
reviews = reviews.lower()
all_text = ''.join([c for c in reviews if c not in punctuation])

# split by lines and spaces
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)

# create word bank
words = all_text.split()

"""
    Encoding words
    we need to pass int into the RNN model to make it easier
"""

# map word to int
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

# dict -> tokenize
reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])

# Test
print("unique words: ", len((vocab_to_int)))

"""
    Encode labels to int
"""
labels_split = labels.split("\n")
encoded_labels = np.array(
    [1 if label == 'positive' else 0 for label in labels_split])

"""
    Remove outliers
"""
# View outlier stats
review_lens = Counter([len(x) for x in reviews_ints])
print("zero-length: {}".format(review_lens[0]))
print("max review: {}".format(max(review_lens)))

# Before
print("num of reviews before: ", len(reviews_ints))

# get indices with reviews of len 0
non_zero_idx = [ii for ii, review in enumerate(
    reviews_ints) if len(review) != 0]

# remvoe 0 len review + labels
reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

# After
print("num of reviews after: ", len(reviews_ints))

"""
    Apply padding
    if short review -> add 0s for padding
    if long review -> truncate 
"""


def pad_features(reviews_ints, seq_length):
    # get features
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    # each review
    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


# Test
seq_length = 200
features = pad_features(reviews_ints, seq_length=seq_length)

assert len(features) == len(
    reviews_ints), "Your features should have as many rows as reviews."
assert len(
    features[0]) == seq_length, "Each feature row should contain seq_length values."

# print first 10 values of the first 30 batches
print(features[:30, :10])

"""
    Split data before training
"""
# Amt to keep for training
split_frac = 0.8

split_idx = int(len(features) * split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x) * 0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

# print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

"""
    Create dataloaders
"""
# Libs

# create tensor dataset
train_data = TensorDataset(torch.from_numpy(
    train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

batch_size = 50

# shuffle training data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print("GPU")
else:
    print("CPU")

"""
    Define network
"""
# Lib


class RNN(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(RNN, self).__init__()
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # create embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            n_layers, dropout=drop_prob, batch_first=True)

        # dropout
        self.dropout = nn.Dropout(0.3)

        # linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        batch_size = x.size(0)

        # pass into the embedding and first LSTM
        x = x.long()
        embeds = self.embedding(x)
        # get output from LSTM layer
        lstm_out, hidden = self.lstm(embeds, hidden)

        # stack LSTMs
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # pass through dropout and fc
        out = self.dropout(lstm_out)
        out = self.fc(out)

        # apply sig func
        sig_out = self.sig(out)

        # reshape batch_size
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        return sig_out, hidden

    def init_hidden(self, batch_size):
        # init weights to zero
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden


# instantiate model with hyperparams
vocab_size = len(vocab_to_int) + 1
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2
lr = 0.001

net = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

# create loss and optimizer func
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

"""
    Train network
"""
epochs = 4

counter = 0  # compared against print_every
print_every = 100  # display every 100
clip = 5  # gradient clipping

if train_on_gpu:
    net.cuda()

net.train()

for e in range(epochs):
    h = net.init_hidden(batch_size)

    # loop through batches
    for inputs, labels in train_loader:
        counter += 1

        if train_on_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()

        # create new var for hidden states
        h = tuple([each.data for each in h])

        # zero any optimized var
        net.zero_grad()

        # get output
        output, h = net(inputs, h)

        # calc loss
        loss = criterion(output.squeeze(), labels.float())

        # backprop
        loss.backward()

        # prevent exploding gradient problem
        nn.utils.clip_grad_norm_(net.parameters(), clip)

        # update weights
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()

            for inputs, labels in valid_loader:
                val_h = tuple([each.data for each in val_h])

                if train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            # after validating set back to train
            net.train()
            print("Epoch: {}/{}...".format(e+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

"""
    Test
"""
test_losses = []
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()

# iter over test data
for inputs, labels in test_loader:
    h = tuple([each.data for each in h])

    if train_on_gpu:
        inputs, labels = inputs.cuda(), labels.cuda()

    output, h = net(inputs, h)

    # calc loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert probs to predicted class (0 or 1)
    pred = torch.round(output.squeeze())

    # compare preds to actual
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
        correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))

# negative test review
test_reviews = 'The best movie I have seen; acting was amazing, best movie of all time so good. Although the acting was a bad, I still liked it'

"""
    Tokenize review
"""


def tokenize_review(test_review):
    test_review = test_review.lower()

    test_text = ''.join([c for c in test_review if c not in punctuation])

    # split by space
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])

    return test_ints


# Add hyperparams
test_ints = tokenize_review(test_reviews)
seq_length = 200
features = pad_features(test_ints, seq_length)
feature_tensor = torch.from_numpy(features)

"""
    Create predict func
"""


def predict(net, test_review, sequence_length=200):
    net.eval()

    # tokenize review
    test_ints = tokenize_review(test_review)

    # pad tokenized sequence
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    if(train_on_gpu):
        feature_tensor = feature_tensor.cuda()

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if(pred.item() == 1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")


"""
    Test 
"""
seq_length = 200  # keep trained and predict length the same

predict(net, test_reviews, seq_length)
