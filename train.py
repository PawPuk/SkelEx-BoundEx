import torch
from torch import nn
from torch.utils.data import DataLoader


class TrainedNeuralNetwork:
    def __init__(self, neural_network, datasets, number_of_parameters, batch_size=32, loss_fn=nn.CrossEntropyLoss(),
                 lr=1e-3, epochs=1, wd=0, opt='Adam', mode='None'):
        self.input_nn = neural_network
        self.train_dataset, self.test_dataset = datasets
        self.batch_size = batch_size
        self.loss_fn = loss_fn
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.number_of_parameters = number_of_parameters
        self.optimizer = opt
        self.mode = mode

    def train(self, dataloader, model, optimizer):
        model.train()
        for batch, batched_samples in enumerate(dataloader):
            if self.mode == 'MNIST':
                print(batched_samples)
                X, y = batched_samples
            else:
                X, y = torch.split(batched_samples, self.number_of_parameters, dim=1)
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = model(X.float())
            if self.loss_fn == nn.BCELoss:
                loss = self.loss_fn()
                loss = loss(torch.argmax(pred, dim=1), y.squeeze(1).long())
            else:
                if self.mode == 'MNIST':
                    loss = self.loss_fn(pred, y)
                else:
                    loss = self.loss_fn(pred, y.squeeze(1).long())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def test(self, dataloader, model):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for batched_samples in dataloader:
                if self.mode == 'MNIST':
                    X, y = batched_samples
                else:
                    X, y = torch.split(batched_samples, self.number_of_parameters, dim=1)
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X.float())
                if self.mode == 'MNIST':
                    test_loss += self.loss_fn(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
                else:
                    test_loss += self.loss_fn(pred, y.squeeze(1).long()).item()
                    correct += (pred.argmax(1).unsqueeze(1) == y).sum().item()
        test_loss /= num_batches
        correct /= size
        # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def main(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True)

        model = self.input_nn.to(self.device)
        if self.optimizer == 'ADAM':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.wd)
        elif self.optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            raise 'Wrong optimizer; Only options available are "ADAM" and "SGD"'

        for t in range(self.epochs):
            # print(f"Epoch {t+1}\n-------------------------------")
            self.train(train_dataloader, model, optimizer)
            self.test(test_dataloader, model)
        print("Done!")

        return model
