import torch
torch.manual_seed(42)
from torch import nn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, activation_function):
        super().__init__()
        num_of_classes = 10
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.output = nn.Linear(hidden_sizes[1], num_of_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.activation_function = activation_function
    
    def forward(self, x):
        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.fc2(x))
        out = self.logsoftmax(self.output(x))
        return out

class Model():
    def __init__(self, epochs, trainloader, testloader, classes):
        self.model = MLP(500, [1000, 300], nn.Sigmoid()).to(device) 
        self.epochs = epochs
        self.criterion = nn.NLLLoss()
        self.optimizer = torch.optim.Adam
        self.trainloader, self.testloader, self.classes = trainloader, testloader, classes

    def fit(self, plot_figs):
        model = self.model
        epochs = self.epochs
        criterion = self.criterion
        optimizer = self.optimizer(model.parameters(), lr=0.0001)
        trainloader = self.trainloader
        testloader = self.testloader

        train_losses = []
        train_accuracies = []

        tic = time.time()
        for epoch in range(epochs):
            running_train_loss = 0
            running_train_accuracy = 0
            print(epoch)
            for train_images, train_labels in trainloader:
                train_images = train_images.cuda()
                train_labels = train_labels.cuda()
                h_train = model(train_images.float())
                predicted_train_labels = torch.max(h_train, dim=1).indices
                running_train_accuracy += torch.mean((predicted_train_labels==train_labels).float()).item()
                optimizer.zero_grad()
                if str(criterion) == "NLLLoss()":
                    train_loss = criterion(h_train, train_labels)
                else:
                    train_loss = criterion(predicted_train_labels, train_labels)
                train_loss.backward()
                optimizer.step()
                running_train_loss += train_loss.item()
            else:
                print("Train Loss is: " 
                      + str(running_train_loss / len(trainloader)))
                train_losses.append(running_train_loss / len(trainloader))
                train_accuracies.append(running_train_accuracy / len(trainloader))
                print("Train accuracy is : " 
                      + str(running_train_accuracy / len(trainloader)))
                print() 
        toc = time.time()
        self.model = model
        if plot_figs:
            self.plot(train_accuracies, train_losses)
        else:
            print("It took " + str(toc-tic) + " seconds to train the model with "+ str(epochs) + " epochs.")

    def predict(self, show_conf_matrix, show_info):
        model = self.model
        testloader = self.testloader
        criterion = self.criterion
        classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        with torch.no_grad():
            test_images, test_labels = next(iter(testloader))
            test_images = test_images.cuda()
            test_labels = test_labels.cuda()
            h_test = model(test_images.float())
            predicted_test_labels = torch.max(h_test, dim=1).indices
            test_accuracy = torch.mean((predicted_test_labels==test_labels).float()).item()
            if str(criterion) == "NLLLoss()":
                loss = criterion(h_test, test_labels).item()
            if str(criterion) == "KLDivLoss()":
                loss = criterion(predicted_test_labels.float(), test_labels.float())
            print("Test loss is: " + str(loss))
            print("Test accurcay is: " + str(test_accuracy))
        
        conf_array = confusion_matrix(test_labels.cpu(), predicted_test_labels.cpu())
        if show_conf_matrix:
            df_cm = pd.DataFrame(conf_array, index = [i for i in classes],
                                             columns = [i for i in classes])
            plt.figure(figsize = (10,10))
            sn.heatmap(df_cm, annot=True)
        
        if show_info:
            recall = np.diag(conf_array) / np.sum(conf_array, axis = 1)
            precision = np.diag(conf_array) / np.sum(conf_array, axis = 0)
            f_score = 2 * (precision * recall) / (precision + recall)
            print("For this model recall is: " + str(recall))
            print("And precision is: " + str(precision))
            print("And f1 score is: " +str(f_score))

        return test_accuracy


    def plot(self, train_accuracies, train_losses):
        epochs = self.epochs

        plt.figure(figsize=(10, 5))
        plt.xlabel("Num of Epochs")
        plt.ylabel("Loss")
        plt.plot(range(epochs), train_losses, linewidth = 2, label = "train loss")
        plt.legend()
        plt.title("Loss based on num of epochs for " + str(epochs) + " epochs")
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.xlabel("Num of Epochs")
        plt.ylabel("Accuracy")
        plt.plot(range(epochs), train_accuracies, linewidth = 2, label = "train accuracies")
        plt.legend()
        plt.title("Accuracies based on num of epochs for " + str(epochs) + " epochs")
        plt.show()

class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
      super().__init__()
      self.fc1 = nn.Linear(input_size, hidden_size)
      self.fc2 = nn.Linear(hidden_size, output_size)
      self.sigmoid = nn.Sigmoid()
      self.relu = nn.ReLU()

  def forward(self, x):
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      x = self.sigmoid(x)
      return x

class Decoder(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
      super().__init__()
      self.fc1 = nn.Linear(input_size, hidden_size)
      self.fc2 = nn.Linear(hidden_size, output_size)
      self.sigmoid = nn.Sigmoid()
      self.relu = nn.ReLU()

  def forward(self, x):
      x = self.fc1(x)
      x = self.relu(x)
      x = self.fc2(x)
      x = self.sigmoid(x)
      return x

class AutoEncoder(nn.Module):
    def __init__(self, feature_size, hidden_sizes):
        super().__init__()
        self.encoder = Encoder(feature_size, hidden_sizes[0], hidden_sizes[1])
        self.decoder = Decoder(hidden_sizes[1], hidden_sizes[2], feature_size)

    def forward(self, x):
        coded = self.encoder(x)
        reconstructed = self.decoder(coded)
        return reconstructed

transform = transforms.Compose([transforms.ToTensor()])


trainset = datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True,
                                          num_workers=2,
                                          pin_memory=True,)

testset = datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                         shuffle=False,
                                         num_workers=2, 
                                         pin_memory = True)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def cifar_imshow(img, tr):
    if tr:
        npimg = img.detach().numpy()
    else:
        npimg = img
    plt.figure(figsize=(1, 1))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

"""
AutoEncoder Training
"""
autoencoder = AutoEncoder(3072, [1000, 500, 1000]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

epochs = 30
for e in range(epochs):
    print(e)
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1).cuda()
        reconstructed_image = autoencoder(images)
        optimizer.zero_grad()
        loss = criterion(reconstructed_image, images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        with torch.no_grad():
            test_images, test_labels = next(iter(testloader))
            test_images = test_images.view(test_images.shape[0], -1).cuda()
            reconstructed_test_image = autoencoder(test_images)
            loss = criterion(reconstructed_test_image, test_images)
            print("Test loss is: " + str(loss.item()))
        print("Training loss:", running_loss/len(trainloader))
        print()

encoder = autoencoder.encoder
decoder = autoencoder.decoder

coded_train_info = []
for train_image, train_label in trainset:
    train_image = train_image.flatten(start_dim=0).to("cuda")
    with torch.no_grad():
        coded_image = encoder(train_image)
    coded_train_info.append([coded_image, train_label])

coded_train_loader = torch.utils.data.DataLoader(coded_train_info, shuffle=True, batch_size=32)

coded_test_info = []
for test_image, test_label in testset:
    test_image = test_image.flatten(start_dim=0).to("cuda")
    with torch.no_grad():
        coded_image = encoder(test_image)
    coded_test_info.append([coded_image, test_label])

coded_test_loader = torch.utils.data.DataLoader(coded_test_info, shuffle=True, batch_size=len(testset))

model = Model(20, coded_train_loader, coded_test_loader, classes)
model.fit(plot_figs=True)
model.predict(show_conf_matrix=True, show_info=True)

"""
PCA IMPLEMENTATION 
"""
X_train = np.zeros((len(trainset), 32*32*3))
X_test = np.zeros((len(testset), 32*32*3))
i = 0
y_train = []
y_test = []
for train_image, train_label in trainset:
    X_train[i] = train_image.flatten(start_dim=0)
    y_train.append(train_label)
    i += 1
i = 0
for test_image, test_label in testset:
    X_test[i] = test_image.flatten(start_dim=0)
    y_test.append(test_label)
    i += 1

X_train = X_train - np.mean(X_train, axis=0)
X_test = X_test - np.mean(X_test, axis=0)

X_train = X_train.T
X_test = X_test.T
cov_matrix = np.dot(X_train, X_train.T)
eig_values, eig_vectors = np.linalg.eig(cov_matrix)
idx = np.argsort(eig_values)
idx = np.flip(idx)
eig_values = eig_values[idx]
eig_vectors = eig_vectors[:, idx]

U = eig_vectors[:, 0:500]
Z_train = np.dot(U.T, X_train)
Z_test = np.dot(U.T, X_test)

Z_train = torch.from_numpy(Z_train.T)
Z_test = torch.from_numpy(Z_test.T)

pca_train_info = []
for x, y in zip(Z_train, y_train):
    pca_train_info.append([x, y])

pca_train_loader = torch.utils.data.DataLoader(pca_train_info, shuffle=True, batch_size=32)

pca_test_info = []
for x, y in zip(Z_test, y_test):
    pca_test_info.append([x, y])

pca_test_loader = torch.utils.data.DataLoader(pca_test_info, shuffle=True, batch_size=len(testset))

model = Model(20, pca_train_loader, pca_test_loader, classes)
model.fit(plot_figs=True)
model.predict(show_conf_matrix=True, show_info=True)