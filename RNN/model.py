# Import lib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

"""
    Load in the data
"""
with open("data/anna.txt", 'r') as f:
    text = f.read()

"""
    Tokenization --> convert text to int
"""
chars = tuple(set(text)) # Create tuple based on text
int2char = dict(enumerate(chars)) # Take the tuple and convert it to int and store as dict
char2int = {ch: ii for ii, ch in int2char.items()}

# encode text
encoded = np.array([char2int[ch] for ch in text])

"""
    Create one-hot encoded func --> char2int and then to vector
    the vector will have a corresponding int index that is 1 to show correlation
"""
def one_hot_encode(arr, n_labels):
    # init encoded arr
    one_hot = np.zeros((arr.size, n_labels), dtype=np.float32)
    
    # Fill with 1s where appropriate
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1
    
    # Reshape 
    one_hot = one_hot.reshape((*arr.shape, n_labels))
    
    return one_hot

"""
    View one_hot result
"""
test_seq = np.array([[3,5,1]])
one_hot = one_hot_encode(test_seq, 8)

print(one_hot)

# arr = where you get batches from
# batch_size = num of seq per batch
# seq_length = num of encoded chars per seq
def get_batches(arr, batch_size, seq_length): 
    batch_size_total = batch_size * seq_length # N x M

    # total num of batches we can have
    n_batches = len(arr) // batch_size_total
    
    # retain a certain num of seq to make full batches
    arr = arr[:n_batches * batch_size_total]
    
    # reshape to rows
    arr = arr.reshape((batch_size, -1))
    
    # one seq at a time
    for n in range(0, arr.shape[1], seq_length):
        # features
        x = arr[:, n:n+seq_length]
        # target
        y = np.zeros_like(x)
        
        try:
            y[:,:-1], y[:,-1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:,:-1], y[:,-1] = x[:, 1:], arr[:, 0]
        yield x, y
    
"""
    GPU 
"""
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print("CPU")
else:
    print("GPU")

"""
    Create network 
"""
class Net(nn.Module):
    
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.lr = lr
        
        # create char dict
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        
        # define LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers, dropout=drop_prob, batch_first=True)
        
        # create dropout layer 
        self.dropout = nn.Dropout(drop_prob)
        
        # fully-connected layer
        self.fc = nn.Linear(n_hidden, len(self.chars))
    
    def forward(self, x, hidden):
        # get input and hidden state and pass 
        r_output, hidden = self.lstm(x, hidden)
        
        # pass to dropout 
        out = self.dropout(r_output)
        
        # stack the LSTM
        out = out.contiguous().view(-1, self.n_hidden)
        
        # pass to fc 
        out = self.fc(out)
        
        return out, hidden 

    def init_hidden(self, batch_size):
        # init weights to zero 
        weight = next(self.parameters()).data
        
        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                     weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                     weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        return hidden

"""
    Train model
    
    val_frac -> amt of data reserved for validation
    clip -> gradient clipping
    print_every -> console the trainin and valid loss
"""
def train(net, data, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    net.train()
    
    # loss and optimizer func
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    
    # create training and validation data
    val_idx = int(len(data) * (1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]
    
    if train_on_gpu:
        net.cuda()
        
    counter = 0 # Compare against print_every to see when to display
    n_chars = len(net.chars)
    for e in range(epochs):
        # init hidden state
        h = net.init_hidden(batch_size)
        
        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1
            
            # one-hot encode and make tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if train_on_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            
            # create new var for hidden state -> if not: use backprop to go through training history
            h = tuple([each.data for each in h])
            
            # zero any optimized var
            net.zero_grad()
            
            # get output
            output, h = net(inputs, h)
            
            # calc loss 
            loss = criterion(output, targets.view(batch_size*seq_length).long())
            
            # backprop
            loss.backward()
            
            # prevent exploding gradient problem
            nn.utils.clip_grad_norm(net.parameters(), clip)
            
            # update weights
            opt.step()
            
            # loss stats
            if counter % print_every == 0:
                # get valid loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                
                for x, y in get_batches(val_data, batch_size, seq_length):
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)
                    
                    val_h = tuple([each.data for each in val_h])
                    
                    inputs, targets = x,y
                    if train_on_gpu:
                        inputs, targets = inputs.cuda(), targets.cuda()
                    
                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length).long())
                    
                    val_losses.append(val_loss.item())
                    
                net.train() # After validating go back to training
                
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))                

"""
    Instantiating model
"""
n_hidden = 512
n_layers = 2

net = Net(chars, n_hidden, n_layers)

batch_size = 128
seq_length = 100
n_epochs = 20

# Train
train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)

"""
    Load checkpoint
"""
model_name = 'rnn_20_epoch.net'

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict(),
              'tokens': net.chars}

with open(model_name, 'wb') as f:
    torch.save(checkpoint, f)

"""
    Predict
"""
def predict(net, char, h=None, top_k=None):
    x = np.array([[net.char2int[char]]])
    x = one_hot_encode(x, len(net.chars))
    inputs = torch.from_numpy(x)
    
    if train_on_gpu:
        inputs = inputs.cuda()
    
    h = tuple([each.data for each in h])
    
    out, h = net(inputs, h)
    
    # get char probs
    p = F.softmax(out, dim=1).data
    
    if train_on_gpu:
        p = p.cpu # Move to CPU instead
    
    # get chars with highest prob
    if top_k is None:
        top_ch = np.arange(len(net.chars))
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()
    
    # select next char w/ rng
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p/p.sum())
    
    return net.int2char[char],h

"""
    Generating sample text
"""
def sample(net, size, prime="The", top_k=None):
    if train_on_gpu:
        net.cuda()
    else:
        net.cpu()
    
    net.eval()
    
    # First iter -> run through all prime chars
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)
    
    chars.append(char)
    
    # Use prev char to get next char
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)
    
    return ''.join(chars)
    
print(sample(net, 1000, prime="Anna", top_k=5))