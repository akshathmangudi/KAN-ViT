import os
import torch
import datetime
import argparse
import logging
import matplotlib.pyplot as plt
from torch.optim import Adam
from tqdm import tqdm, trange
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from model import VisionTransformer
from utils import calculate_metrics, save_metrics, setup_logging, plot_all_curves


def main(train_loader, test_loader, args):
    device = torch.device(args.device)
    logging.info(f"Using device: {device}" +
                 (f" ({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else ""))

    mnist_model = VisionTransformer(
        chw=(1, 28, 28),
        n_patches=7,
        n_blocks=args.n_blocks,
        d_hidden=args.d_hidden,
        n_heads=args.n_heads,
        out_d=10,
        type=args.model_type
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(mnist_model.parameters(), lr=args.learning_rate)

    metrics_log_filename = setup_logging(args.log_dir)

    # For plotting
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    train_balaccs, test_balaccs = [], []
    train_f1s, test_f1s = [], []
    train_aucs, test_aucs = [], []

    epochs_list = list(range(1, args.epochs + 1))

    for epoch in trange(args.epochs, desc="Train Epochs"):
        mnist_model.train()
        train_loss_epoch = 0.0
        y_true_train, y_pred_train, y_pred_proba_train = [], [], []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} [train]", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            y_hat = mnist_model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.detach().cpu().item()

            y_true_train.extend(y.cpu().numpy())
            y_pred_train.extend(torch.argmax(y_hat, dim=1).cpu().numpy())
            y_pred_proba_train.extend(torch.nn.functional.softmax(y_hat, dim=1).detach().cpu().numpy())

        train_loss_epoch /= len(train_loader)
        accuracy, balanced_accuracy, f1, roc_auc = calculate_metrics(
            y_true_train, y_pred_train, y_pred_proba_train
        )

        logging.info(f"Epoch {epoch + 1}/{args.epochs} => TRAIN")
        logging.info(f"  Loss: {train_loss_epoch:.4f}, Acc: {accuracy:.4f}, BalAcc: {balanced_accuracy:.4f}, "
                     f"F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

        # Save train metrics to file
        save_metrics(metrics_log_filename, epoch + 1, "Train",
                     train_loss_epoch, accuracy, balanced_accuracy, f1, roc_auc, flag=0)

        # store for plotting
        train_losses.append(train_loss_epoch)
        train_accs.append(accuracy)
        train_balaccs.append(balanced_accuracy)
        train_f1s.append(f1)
        train_aucs.append(roc_auc)

        # --- Evaluate on test set ---
        mnist_model.eval()
        with torch.no_grad():
            test_loss_epoch = 0.0
            y_true_test, y_pred_test, y_pred_proba_test = [], [], []

            for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1} [test]", leave=False):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = mnist_model(x)
                loss = criterion(y_hat, y)
                test_loss_epoch += loss.detach().cpu().item()

                y_true_test.extend(y.cpu().numpy())
                y_pred_test.extend(torch.argmax(y_hat, dim=1).cpu().numpy())
                y_pred_proba_test.extend(torch.nn.functional.softmax(y_hat, dim=1).cpu().numpy())

            test_loss_epoch /= len(test_loader)
            accuracy_test, balanced_accuracy_test, f1_test, roc_auc_test = calculate_metrics(
                y_true_test, y_pred_test, y_pred_proba_test
            )

            logging.info(f"Epoch {epoch + 1}/{args.epochs} => TEST")
            logging.info(f"  Loss: {test_loss_epoch:.4f}, Acc: {accuracy_test:.4f}, BalAcc: {balanced_accuracy_test:.4f}, "
                         f"F1: {f1_test:.4f}, ROC-AUC: {roc_auc_test:.4f}")

            save_metrics(metrics_log_filename, epoch + 1, "Test",
                         test_loss_epoch, accuracy_test, balanced_accuracy_test, f1_test, roc_auc_test, flag=1)

            test_losses.append(test_loss_epoch)
            test_accs.append(accuracy_test)
            test_balaccs.append(balanced_accuracy_test)
            test_f1s.append(f1_test)
            test_aucs.append(roc_auc_test)

    # After all epochs, generate plots
    plot_all_curves(
        epochs_list,
        train_losses, test_losses,
        train_accs, test_accs,
        train_balaccs, test_balaccs,
        train_f1s, test_f1s,
        train_aucs, test_aucs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark KAN-based Vision Transformers on MNIST')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate for optimizer')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use for training (cuda/cpu)')
    parser.add_argument('--model-type', type=str, default='vanilla',
                        help='variant: [vanilla, efficientkan, sine, cheby, fast, flash-attn, fourier]')
    parser.add_argument('--n-blocks', type=int, default=4,
                        help='number of transformer blocks')
    parser.add_argument('--d-hidden', type=int, default=64,
                        help='hidden dimension of each transformer block')
    parser.add_argument('--n-heads', type=int, default=2,
                        help='number of attention heads')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='directory to store logs')
    args = parser.parse_args()

    transform = transforms.ToTensor()
    train_mnist = MNIST(root='./mnist', train=True,
                        download=True, transform=transform)
    test_mnist = MNIST(root='./mnist', train=False,
                       download=True, transform=transform)

    train_loader = DataLoader(train_mnist, shuffle=True, batch_size=args.batch_size)
    test_loader = DataLoader(test_mnist, shuffle=False, batch_size=args.batch_size)

    main(train_loader=train_loader, test_loader=test_loader, args=args)
