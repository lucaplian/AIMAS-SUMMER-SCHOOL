import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from collections import OrderedDict


def plot_admissions(data):
    X = np.array(data[["gre","gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'red', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'cyan', edgecolor = 'k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')

def plot_points(X, y):
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')
    
# Plot a line
def plot_line(m, b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)


def test_network(net, trainloader):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # Create Variables for the inputs and targets
    inputs = Variable(images)
    targets = Variable(images)

    # Clear the gradients from all Variables
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    output = net.forward(inputs)
    loss = criterion(output, targets)
    loss.backward()
    optimizer.step()

    return True


def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

def view_recon(img, recon):
    ''' Function for displaying an image (as a PyTorch Tensor) and its
        reconstruction also a PyTorch Tensor
    '''

    fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True)
    axes[0].imshow(img.numpy().squeeze())
    axes[1].imshow(recon.data.numpy().squeeze())
    for ax in axes:
        ax.axis('off')
        ax.set_adjustable('box-forced')

def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()


def visualize_filters(filters):
    # visualize all four filters
    fig = plt.figure(figsize=(10, 5))
    for i in range(len(filters)):
        ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i], cmap='gray')
        ax.set_title('Filter %s' % str(i+1))
        width, height = filters[i].shape
        for x in range(width):
            for y in range(height):
                ax.annotate(str(filters[i][x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white' if filters[i][x][y]<0 else 'black')


def visualize_filters_outputs(layer, n_filters=4):
    fig = plt.figure(figsize=(20, 20))
    for i in range(n_filters):
        ax = fig.add_subplot(1, n_filters, i+1, xticks=[], yticks=[])
        # grab layer outputs
        ax.imshow(np.squeeze(layer[0,i].data.numpy()), cmap='gray')
        ax.set_title('Output %s' % str(i+1))

# function to save our model
def save_model(model, checkpoint='checkpoint.pth'):
    
    # We first create a model dictionary out of all its children (name, layer) pairs
    checkpoint_model = OrderedDict(model.named_children())

    # We create a dictionary in which we save the model and its state
    checkpoint_dict = {
        'model_': checkpoint_model,
        'state_dict': model.state_dict()
    }
    
    # We save the dictonary a file
    torch.save(checkpoint_dict, checkpoint)

# function to load our model
def load_model(checkpoint='checkpoint.pth'):
    # We first load the model from the file
    checkpoint_dict = torch.load(checkpoint)
    
    # We use an nn.Sequential object to re-create our model using the dictionary
    model = nn.Sequential(checkpoint_dict['model_'])
    
    # We can now load the state in our newly created model
    model.load_state_dict(checkpoint_dict['state_dict'])
    
    return model


# PyTorch nn does not include a flattening operation but we can easly define one
class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


# Some of the networks used in the class notebooks
class FCNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)


class MLCINet(nn.Module):
    def __init__(self):
        super(MLCINet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1), # (1x28x28) -> (64x28x28)
            nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1), # (64x28x28) -> (128x28x28)
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),  # (128x28x28) -> (128x14x14)

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1), # (128x14x14) -> (128x14x14)            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.BatchNorm2d(num_features=128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1), # (128x14x14) -> (256x14x14)
            nn.BatchNorm2d(num_features=256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2), # (256x14x14) -> (256x7x7)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            View(shape=(-1, 256 * 7 * 7)),
            nn.Linear(in_features=256 * 7 * 7, out_features=1024),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1024, out_features=10),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MLCISimpleNet(nn.Module):
    def __init__(self):
        super(MLCISimpleNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )

        self.classifier = nn.Sequential(
            View(shape=(-1, 12 * 4 * 4)),
            nn.Linear(in_features=12 * 4 * 4, out_features=120),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=120, out_features=60),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=60, out_features=10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



# Basic training and validation loops
def train(model, trainloader, testloader, criterion, optimizer, epochs=5, print_every=100):
    
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1
            
            # Flatten images into a 784 long vector
            images.resize_(images.size()[0], 784)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()

                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

                running_loss = 0

                # Make sure dropout and grads are on for training
                model.train()
                
                
def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:

        images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy
