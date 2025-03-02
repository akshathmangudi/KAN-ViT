import os
import math
import torch
import datetime
import logging
import matplotlib.pyplot as plt
from torch import einsum
from einops import rearrange
from torch.autograd.function import Function
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

EPSILON = 1e-10


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate the accuracy, balanced accuracy, F1 score, and ROC AUC score 
    between the true labels and the predicted labels.
    """
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    # handle multi-class
    y_true_bin = torch.nn.functional.one_hot(
        torch.tensor(y_true), num_classes=10).numpy()
    roc_auc = roc_auc_score(y_true_bin, y_pred_proba,
                            average='weighted', multi_class='ovr')
    return accuracy, balanced_accuracy, f1, roc_auc


def save_metrics(filename, epoch, phase, loss, accuracy, balanced_accuracy, f1, roc_auc, flag):
    """
    Save training or test metrics to a log file. If flag == 0 => training; else => test.
    """
    os.makedirs('logs', exist_ok=True)
    with open(filename, 'a') as f:
        if flag == 0:
            f.write(f"Epoch: {epoch}, Phase: {phase}\n")
            f.write(f"  Loss: {loss:.4f}\n")
            f.write(f"  Accuracy: {accuracy:.4f}\n")
            f.write(f"  Balanced Accuracy: {balanced_accuracy:.4f}\n")
            f.write(f"  F1 Score: {f1:.4f}\n")
            f.write(f"  ROC AUC: {roc_auc:.4f}\n\n")
        else:
            f.write(f"Phase: {phase}\n")
            f.write(f"  Loss: {loss:.4f}\n")
            f.write(f"  Accuracy: {accuracy:.4f}\n")
            f.write(f"  Balanced Accuracy: {balanced_accuracy:.4f}\n")
            f.write(f"  F1 Score: {f1:.4f}\n")
            f.write(f"  ROC AUC: {roc_auc:.4f}\n\n")


def setup_logging(log_dir='logs'):
    """
    Set up logging configuration to file + console.
    Returns path to the plain-text metrics file.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_filename = os.path.join(log_dir, f'training_{timestamp}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    # We'll store metrics in a separate text file for clarity
    return os.path.join(log_dir, f'mnist_metrics_{timestamp}.txt')


def plot_all_curves(epochs_list,
                    train_losses, test_losses,
                    train_accs, test_accs,
                    train_balaccs, test_balaccs,
                    train_f1s, test_f1s,
                    train_aucs, test_aucs):
    """
    Generate and save plots for loss, accuracy, balanced accuracy, F1 score, and ROC AUC.
    """
    os.makedirs("plots", exist_ok=True)

    # 1) Loss
    plt.figure()
    plt.plot(epochs_list, train_losses, label="Train Loss")
    plt.plot(epochs_list, test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs. Epoch")
    plt.legend()
    plt.savefig("plots/loss_vs_epoch.png")
    plt.close()

    # 2) Accuracy
    plt.figure()
    plt.plot(epochs_list, train_accs, label="Train Accuracy")
    plt.plot(epochs_list, test_accs, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Epoch")
    plt.legend()
    plt.savefig("plots/accuracy_vs_epoch.png")
    plt.close()

    # 3) Balanced Accuracy
    plt.figure()
    plt.plot(epochs_list, train_balaccs, label="Train Balanced Acc")
    plt.plot(epochs_list, test_balaccs, label="Test Balanced Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Balanced Accuracy")
    plt.title("Balanced Accuracy vs. Epoch")
    plt.legend()
    plt.savefig("plots/balanced_acc_vs_epoch.png")
    plt.close()

    # 4) F1 Score
    plt.figure()
    plt.plot(epochs_list, train_f1s, label="Train F1")
    plt.plot(epochs_list, test_f1s, label="Test F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs. Epoch")
    plt.legend()
    plt.savefig("plots/f1_vs_epoch.png")
    plt.close()

    # 5) ROC AUC
    plt.figure()
    plt.plot(epochs_list, train_aucs, label="Train ROC AUC")
    plt.plot(epochs_list, test_aucs, label="Test ROC AUC")
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.title("ROC AUC vs. Epoch")
    plt.legend()
    plt.savefig("plots/rocauc_vs_epoch.png")
    plt.close()

