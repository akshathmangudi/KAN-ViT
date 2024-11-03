import os
import torch
import datetime
import argparse
import logging
from torch.optim import Adam
from tqdm import tqdm, trange
from torchvision import transforms
from model import VisionTransformer
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from utils import calculate_metrics, save_metrics, setup_logging


def main(train_loader, test_loader, args):
    flag = 0
    device = torch.device(args.device)
    logging.info(f"Using device: {device} " +
                 (f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else ""))

    mnist_model = VisionTransformer(
        (1, 28, 28), n_patches=7, n_blocks=2, d_hidden=8, n_heads=2, out_d=10,
        type=args.model_type).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(mnist_model.parameters(), lr=args.learning_rate)

    metrics_log_filename = setup_logging(args.log_dir)

    for epoch in trange(args.epochs, desc="train"):
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

        accuracy, balanced_accuracy, f1, roc_auc = calculate_metrics(
            y_true_train, y_pred_train, y_pred_proba_train)

        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        logging.info(f"  Train Loss: {train_loss:.4f}")
        logging.info(f"  Train Accuracy: {accuracy:.4f}")
        logging.info(f"  Train Balanced Accuracy: {balanced_accuracy:.4f}")
        logging.info(f"  Train F1 Score: {f1:.4f}")
        logging.info(f"  Train ROC AUC: {roc_auc:.4f}")

        if epoch == args.epochs - 1:
            save_metrics(metrics_log_filename, epoch + 1, "Train",
                         train_loss, accuracy, balanced_accuracy, f1, roc_auc, flag)

    # Testing
    mnist_model.eval()
    with torch.no_grad():
        flag = 1
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

        accuracy, balanced_accuracy, f1, roc_auc = calculate_metrics(
            y_true_test, y_pred_test, y_pred_proba_test)

        logging.info("Test Results:")
        logging.info(f"  Test Loss: {test_loss:.4f}")
        logging.info(f"  Test Accuracy: {accuracy:.4f}")
        logging.info(f"  Test Balanced Accuracy: {balanced_accuracy:.4f}")
        logging.info(f"  Test F1 Score: {f1:.4f}")
        logging.info(f"  Test ROC AUC: {roc_auc:.4f}")

        save_metrics(metrics_log_filename, args.epochs, "Test", test_loss,
                     accuracy, balanced_accuracy, f1, roc_auc, flag)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='The official repository to benchmark KAN-based Vision Transformers')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate for optimizer')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use for training (cuda/cpu)')
    parser.add_argument('--model-type', type=str, default='vanilla',
                        help='the variant to run, options: [vanilla, efficient-kan, fast-kan, fourier-kan, sine-kan, cheby-kan]')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='directory to store logs')
    args = parser.parse_args()

    transform = transforms.ToTensor()
    train_mnist = MNIST(root='./mnist', train=True,
                        download=True, transform=transform)
    test_mnist = MNIST(root='./mnist', train=False,
                       download=True, transform=transform)

    train_loader = DataLoader(
        train_mnist, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_mnist, shuffle=False,
                             batch_size=args.batch_size)

    main(train_loader=train_loader,
         test_loader=test_loader,
         args=args)
