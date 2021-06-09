import torch
import numpy as np


def train(cnn_model, device, iterator, optimizer, criterion, progress_bar):

    epoch_loss = 0.0
    cnn_model.to(device)
    cnn_model.train()

    for step, (img, target) in enumerate(iterator):
        img = img.to(device)
        target = target.type(torch.LongTensor)
        target = target.to(device)

        # Forward pass of the CNN
        y = cnn_model(img)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # calculate the loss
        loss = criterion(y, target)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()

        # perform a single optimization step (parameter update)
        optimizer.step()

        # Save epoch loss and epoch accuracy
        epoch_loss += loss.item()

        progress_bar.value = step

    return epoch_loss / (iterator.batch_size * len(iterator))


def evaluate(cnn_model, device, iterator, criterion, progress_bar):
    epoch_loss = 0
    correct_classifications = 0

    cnn_model.to(device)
    cnn_model.eval()

    with torch.no_grad():
        for step, (img, target) in enumerate(iterator):

            img = img.to(device)

            target = target.type(torch.LongTensor)
            target = target.to(device)

            # forward pass of the CNN
            y = cnn_model(img)

            # calculate the loss
            loss = criterion(y, target)

            correct_classifications += np.equal(list(map(lambda x: torch.argmax(x), y.cpu())), list(target.cpu())).sum()

            # Save epoch loss
            epoch_loss += loss.item()

            progress_bar.value = step

    return (epoch_loss / (iterator.batch_size * len(iterator))), (
        correct_classifications / float(iterator.batch_size * len(iterator))
    )


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
