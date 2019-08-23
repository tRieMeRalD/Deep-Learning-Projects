# Import libs
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from torchvision import datasets
import torch
import numpy as np

train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print("GPU")

    """
    Load in datasets
"""
# Import libs

num_workers = 0  # No. of subprocess to retrieve data
batch_size = 20  # 20 data points per batch
valid_size = 0.2

# Change data to float tensor
transform = transforms.ToTensor()

# Specify training and test datasets
train_data = datasets.MNIST(root="data", train=True,
                            download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False,
                           download=True, transform=transform)

# Randomize and set data for validation
num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

# Define train and valid batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# Prep data loader
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, num_workers=num_workers)

"""
    Visualize data
"""
get_ipython().run_line_magic('matplotlib', 'inline')

# Show one batch
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# Plot images with correct label
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap="gray")
    ax.set_title(str(labels[idx].item()))

"""
    Show normalized version
"""
img = np.squeeze(images[0])

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111)
ax.imshow(img, cmap="gray")
width, height = img.shape
thresh = img.max() / 2.5

for x in range(width):
    for y in range(height):
        val = round(img[x][y], 2) if img[x][y] != 0 else 0
        ax.annotate(str(val), xy=(y, x),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if img[x][y] < thresh else 'black')

"""
    Create network
"""
# Import libs

# Define network


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Define hidden layers
        hidden_1, hidden_2 = 512, 512

        # Define fully-connected layers
        self.fc1 = nn.Linear(28 * 28, hidden_1)  # In, out
        self.fc2 = nn.Linear(hidden_1, hidden_2)  # In , out
        self.fc3 = nn.Linear(hidden_2, 10)  # 10 classes for out

        # Add dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Flatten img
        x = x.view(-1, 28*28)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)
        return x


model = Net()
print(model)

if train_on_gpu:
    model.cuda()


"""
    Create loss and optimizer func
"""

# Loss func
criterion = nn.CrossEntropyLoss()

# Optimizer func
optimizer = optim.SGD(model.parameters(), lr=0.01)

"""
    Train / validate network
"""
n_epochs = 50  # No. of times to train

valid_loss_min = np.Inf  # Compares against valid_loss

for epoch in range(1, n_epochs+1):
    # Counter var
    train_loss = 0.0
    valid_loss = 0.0

    # Train
    model.train()
    for data, target in train_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        # Clear gradient for optimized var
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Calc loss
        loss = criterion(output, target)

        # Backprop
        loss.backward()

        # Update weights
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    # validate model
    model.eval()
    for data, target in valid_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)

    # Calc avg loss
    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch+1,
        train_loss,
        valid_loss
    ))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'model.pt')
        valid_loss_min = valid_loss

"""
    Test network
"""
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
for data, target in test_loader:
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()

    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item() * data.size(0)

    # Compare pred to actual
    _, pred = torch.max(output, 1)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(
        correct_tensor.cpu().numpy())

    # calculate test accuracy for each object class
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

# Calc avg acc
test_loss = test_loss / len(test_loader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            str(i), 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' %
              (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

"""
    Visualize output results
"""
# One batch
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

if train_on_gpu:
    images = images.cuda()

output = model(images)
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(
    preds_tensor.cpu().numpy())

fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images.cpu()[idx]), cmap='gray')
    ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                 color=("green" if preds[idx] == labels[idx] else "red"))
