import os
import torch
import datetime
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm, trange
from model import VisionTransformer


def calculate_metrics(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    y_true_bin = torch.nn.functional.one_hot(
        torch.tensor(y_true), num_classes=10).numpy()
    roc_auc = roc_auc_score(y_true_bin, y_pred_proba,
                            average='weighted', multi_class='ovr')

    return accuracy, balanced_accuracy, f1, roc_auc


def save_metrics(filename, epoch, phase, loss, accuracy, balanced_accuracy, f1, roc_auc):
    os.makedirs('logs', exist_ok=True)
    with open(f'logs/{filename}', 'a') as f:
        f.write(f"Epoch: {epoch}, Phase: {phase}\n")
        f.write(f"  Loss: {loss:.4f}\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n")
        f.write(f"  Balanced Accuracy: {balanced_accuracy:.4f}\n")
        f.write(f"  F1 Score: {f1:.4f}\n")
        f.write(f"  ROC AUC: {roc_auc:.4f}\n\n")


def main(train_loader, test_loader, epochs: int, device, type: str = "vanilla"):
    print("Using device: ", device,
          f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    mnist_model = VisionTransformer(
        (1, 28, 28), n_patches=7, n_blocks=2, d_hidden=8, n_heads=2, out_d=10, type=type).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(mnist_model.parameters(), lr=0.001)

    # Create a unique filename for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"mnist_metrics_{timestamp}.txt"

    for epoch in trange(epochs, desc="train"):
        train_loss = 0.0
        y_true_train, y_pred_train, y_pred_proba_train = [], [], []

        mnist_model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = mnist_model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_true_train.extend(y.cpu().numpy())
            y_pred_train.extend(torch.argmax(y_hat, dim=1).cpu().numpy())
            y_pred_proba_train.extend(torch.nn.functional.softmax(
                y_hat, dim=1).detach().cpu().numpy())

        # Calculate training metrics
        accuracy, balanced_accuracy, f1, roc_auc = calculate_metrics(
            y_true_train, y_pred_train, y_pred_proba_train)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Accuracy: {accuracy:.4f}")
        print(f"  Train Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"  Train F1 Score: {f1:.4f}")
        print(f"  Train ROC AUC: {roc_auc:.4f}")

        # Save metrics for the last epoch
        if epoch == epochs - 1:
            save_metrics(log_filename, epoch + 1, "Train",
                         train_loss, accuracy, balanced_accuracy, f1, roc_auc)

    # Testing
    mnist_model.eval()
    with torch.no_grad():
        test_loss = 0.0
        y_true_test, y_pred_test, y_pred_proba_test = [], [], []

        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = mnist_model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            y_true_test.extend(y.cpu().numpy())
            y_pred_test.extend(torch.argmax(y_hat, dim=1).cpu().numpy())
            y_pred_proba_test.extend(
                torch.nn.functional.softmax(y_hat, dim=1).cpu().numpy())

        # Calculate test metrics
        accuracy, balanced_accuracy, f1, roc_auc = calculate_metrics(
            y_true_test, y_pred_test, y_pred_proba_test)

        print("Test Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test Balanced Accuracy: {balanced_accuracy:.4f}")
        print(f"  Test F1 Score: {f1:.4f}")
        print(f"  Test ROC AUC: {roc_auc:.4f}")

        # Save test metrics
        save_metrics(log_filename, epochs, "Test", test_loss,
                     accuracy, balanced_accuracy, f1, roc_auc)


if __name__ == "__main__":
    transform = transforms.ToTensor()
    train_mnist = MNIST(root='./mnist', train=True,
                        download=True, transform=transform)
    test_mnist = MNIST(root='./mnist', train=False,
                       download=True, transform=transform)
    train_loader = DataLoader(train_mnist, shuffle=True, batch_size=128)
    test_loader = DataLoader(test_mnist, shuffle=False, batch_size=128)
    main(train_loader=train_loader,
         test_loader=test_loader, epochs=5, device="cuda", type="vanilla")