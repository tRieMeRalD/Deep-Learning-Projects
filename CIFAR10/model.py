# import libraries
import torch 
import numpy as np

# check if GPU is avail for training
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print("Training on CPU")
else:
    print("Training on GPU")

"""
   Loading in the dataset
"""
# import lib to modify dataset 
from torchvision import datasets
import torchvision.transforms as transforms 
from torch.utils.data.sampler import SubsetRandomSampler

# At most have 1 worker retrieve data 
num_workers = 0
# Samples to load per batch that is trained
batch_size = 20
# Set aside 20% of training set for validation
valid_size = 0.2

# Convert data to normalized float tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # ((mean), (standard deviation)) <-- precomputed
])

# define training and test datasets
train_data = datasets.CIFAR10("data", train=True, download=True, transform=transform)
test_data = datasets.CIFAR10("data", train=False, download=True, transform=transform)

# Validation indices
num_train = len(train_data) # gets length of train_data
indices = list(range(num_train)) # turns length num into a list
np.random.shuffle(indices) # randomizes the sequence of the items in list
split = int(np.floor(valid_size * num_train)) # splits the list into 2 components
train_idx, valid_idx = indices[split:], indices[:split] # assigns the 2 components to train and validation

# Define sampler for training and validation batches --> so that it knows which part of the batch to be train or valid
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Prepare data loaders by combining dataset and sampler 
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

# Specify class outputs 
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

"""
   Visualize our dataset
"""
# Import lib to visualize
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

# create helper func to unnormalize and display img
def imgshow(img):
    img = img / 2 + 0.5 # To unnormalize
    plt.imshow(np.transpose(img, (1,2,0)))

# Get one batch of training img
dataiter = iter(train_loader)
images, labels = dataiter.next() # Get the img and label values
images = images.numpy() # Convert img to numpy for display 

# plot img from batch with its labels
fig = plt.figure(figsize=(25,4))

# display 20 images
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imgshow(images[idx])
    ax.set_title(classes[labels[idx]])

"""
   View an image in more detail 
"""
rgb_img = np.squeeze(images[3])
channels = ['red channel', 'green channel', 'blue channel'] # separate into 3 different channels

fig = plt.figure(figsize=(36,36)) # dimensions of plot
for idx in np.arange(rgb_img.shape[0]):
    ax = fig.add_subplot(1,3,idx+1) # 1 x 3 grid, idx+1 subplot
    img = rgb_img[idx]
    ax.imshow(img, cmap='gray') # show in gray
    ax.set_title(channels[idx]) # channel titles
    width, height = img.shape # get width, height values from img
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y]) if img[x][y] != 0 else 0
            ax.annotate(str(val), xy=(y,x), 
                       horizontalalignment='center', 
                       verticalalignment='center',
                       size=8,
                       color="white" if img[x][y]<thresh else 'black')

"""
   Define network architecture
"""
# Import lib for network
import torch.nn as nn
import torch.nn.functional as F

# Define architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        # Defining conv layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) # 3 in-channels, 16 out-channels, kernal size of 3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1) # 16 in, 32 out, kernal of 3
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1) # 32 in, 64 out, kernal of 3
        
        # Define max pooling layer
        self.pool = nn.MaxPool2d(2,2) # kernal size, stride
        
        # Define fully connected layers
        self.fc1 = nn.Linear(64*4*4, 500) # 64 comes from the conv3 output
        self.fc2 = nn.Linear(500, 10) # 10 out for the 10 classes
        
        # Define dropout to prevent overfitting
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Create seq. of conv layers followed by pooling layers with ReLU activation func
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the img from the last output from conv3 in this case
        x = x.view(-1, 64*4*4)
        
        # Apply dropout 
        x = self.dropout(x)
        
        # Apply activation func
        x = F.relu(self.fc1(x))
        x = self.dropout(x) # Apply dropout
        x = self.fc2(x) # Output
        
        return x

# Create out CNN 
model = Net()
print(model)

# Train model on GPU if avail
if train_on_gpu:
    model.cuda()    

"""
   Create our loss and optimize func to update weights and bias
"""
# Import libs
import torch.optim as optim

# Loss func
criterion = nn.CrossEntropyLoss() # Takes the error and applies a -log func

# Optimizer func
optimizer = optim.SGD(model.parameters(), lr=0.01)

"""
   Train the network
"""
# No. of rounds to train the networks
n_epochs = 30

valid_loss_min = np.Inf # Infinity --> used to compare against valid_loss

for epoch in range(1, n_epochs+1):
    # Define counters
    train_loss = 0.0
    valid_loss = 0.0
    
    # Training the model 
    model.train()
    for data, target in train_loader: 
        # Train model on GPU if avail
        if train_on_gpu:
            data, target = data.cuda(), target.cuda() 
        
        # Clear gradient for all optmized var
        optimizer.zero_grad()
        
        # Forward pass to compute predicted outputs
        output = model(data)
        
        # Calculate the loss func of forward pass
        loss = criterion(output, target) # Compares the predicted to actual 
        
        # Perform back propagation 
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Update training loss counter
        train_loss += loss.item() * data.size(0)
        
    
    # Validate model
    model.eval()
    for data, target in valid_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)
        
    
    # Calc avg losses
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)
    
    # Print stats
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
    epoch, train_loss, valid_loss))
    
    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_cifar.pt')
        valid_loss_min = valid_loss

"""
   Test the network
"""
# Define counter var
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over the test dataset
for data, target in test_loader:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item() * data.size(0)
    
    # Convert probability to predicted class 
    _, pred = torch.max(output, 1)
    
    # Compare pred to actual
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    
    # Calc test acc 
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# Avg test loss
test_loss = test_loss / len(test_loader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))