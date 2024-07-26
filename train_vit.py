import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm, trange
from src.vit import ViT

def main(train_loader, test_loader):
    """
    This code contains the training and testing loop for training the vision transformers model. It requires two
    parameters

    :param train_loader: The dataloader for the training set for training the model.
    :param test_loader: The dataloader for the testing set during evaluation phase.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mnist_model = ViT((1, 28, 28), n_patches=7, n_blocks=2, d_hidden=8, n_heads=2, out_d=10).to(device)
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    epochs = 8
    lr = 0.005

    optimizer = Adam(mnist_model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in trange(epochs, desc="train"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = mnist_model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss:.2f}")

    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = mnist_model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)

        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
        
if __name__ == "__main__": 
    transform = transforms.ToTensor()
    train_mnist = MNIST(root='./mnist', train=True, download=True, transform=transform)
    test_mnist = MNIST(root='./mnist', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_mnist, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_mnist, shuffle=False, batch_size=128)
    main(train_loader=train_loader, test_loader=test_loader)