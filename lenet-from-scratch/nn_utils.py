import torch # pytorch basic package
from torch import nn # neural net 
from torch.utils.data import DataLoader, Dataset # to work with data
from torchvision import datasets # built-in data
from torchvision.transforms import ToTensor # to convert nparrays/images into pytorch tensors

# training procedure for a single epoch
def train(train_dataloader, model, loss_fn, optimizer, logging_frequency, device):
    num_samples = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    model.train() # if we are in training mode, some functions will behave differently (e.g. dropout, batchnorm)

    training_loss = 0

    # iterate through the whole dataset, batch just stores the index of the batch.
    # if we have a total of 128 images, with a batch size of 32 we will have 4 batches before the for loop stops
    for batch, (X,y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # calc yhat and loss value
        yhat = model(X)
        loss = loss_fn(yhat, y)
        training_loss += loss.item()

        # backpropagation: calculate the partial derivatives for each neural net parameter
        loss.backward()
        # update neural network parameters according to the partial derivatives
        optimizer.step()
        # zero out the variables that are used for storing the gradients
        optimizer.zero_grad()

        # we ususally don't want to print information at every batch, so make it a bit more scarce:
        if batch % logging_frequency == 0:
            current_loss = loss.item()
            current = (batch + 1) * len(X) # len(X) is the batch size
            print(f'{current}/{num_samples} : {current_loss = }')
    
    training_loss /= num_batches
    return training_loss

def test(epoch, test_dataloader, model, loss_fn, device):
    num_samples = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    model.eval() # if we are in eval mode, some functions behave differently
    test_loss = 0 
    correct = 0
    with torch.no_grad(): # we do not want to calculate gradients in eval mode
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y).item()
            test_loss += loss # we summarize the test loss across all batches
            correct_pred_locations = (pred.argmax(1) == y).type(torch.float) # also torch.argmax(pred, dim=1) is correct, and then we convert these locations to float
            correct += correct_pred_locations.sum().item() # we summarize how many items we got correct
    test_loss /= num_batches # we average the loss
    correct /= num_samples
    print(f'End of epoch {epoch+1}\n Accuracy: {(100*correct):.2f}%, Average loss: {test_loss:.4f}\n')
    return test_loss