import os
import torch
import argparse
import logging
from torch.optim import Adam
from tqdm import tqdm, trange
from torchvision import transforms
from model import VisionTransformer
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from utils import calculate_metrics, save_metrics, setup_logging

def main(train_loader, test_loader, args):
    device = torch.device(args.device)
    logging.info(f"Using device: {device} " +
                 (f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else ""))

    model = VisionTransformer(
        (3, 32, 32), n_patches=4, n_blocks=args.n_blocks, d_hidden=args.d_hidden, n_heads=args.n_heads, out_d=100,
        type=args.model_type).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    metrics_log_filename = setup_logging(args.log_dir)

    for epoch in trange(args.epochs, desc="train"):
        train_loss = 0.0
        y_true_train, y_pred_train, y_pred_proba_train = [], [], []

        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_true_train.extend(y.cpu().numpy())
            y_pred_train.extend(torch.argmax(y_hat, dim=1).cpu().numpy())
            y_pred_proba_train.extend(torch.nn.functional.softmax(y_hat, dim=1).detach().cpu().numpy())

        accuracy, balanced_accuracy, f1, roc_auc = calculate_metrics(y_true_train, y_pred_train, y_pred_proba_train)

        logging.info(f"Epoch {epoch + 1}/{args.epochs}")
        logging.info(f"  Train Loss: {train_loss:.4f}")
        logging.info(f"  Train Accuracy: {accuracy:.4f}")
        logging.info(f"  Train Balanced Accuracy: {balanced_accuracy:.4f}")
        logging.info(f"  Train F1 Score: {f1:.4f}")
        logging.info(f"  Train ROC AUC: {roc_auc:.4f}")

        if epoch == args.epochs - 1:
            save_metrics(metrics_log_filename, epoch + 1, "Train", train_loss, accuracy, balanced_accuracy, f1, roc_auc, flag=0)

    # Testing
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        y_true_test, y_pred_test, y_pred_proba_test = [], [], []

        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            y_true_test.extend(y.cpu().numpy())
            y_pred_test.extend(torch.argmax(y_hat, dim=1).cpu().numpy())
            y_pred_proba_test.extend(torch.nn.functional.softmax(y_hat, dim=1).cpu().numpy())

        accuracy, balanced_accuracy, f1, roc_auc = calculate_metrics(y_true_test, y_pred_test, y_pred_proba_test)

        logging.info("Test Results:")
        logging.info(f"  Test Loss: {test_loss:.4f}")
        logging.info(f"  Test Accuracy: {accuracy:.4f}")
        logging.info(f"  Test Balanced Accuracy: {balanced_accuracy:.4f}")
        logging.info(f"  Test F1 Score: {f1:.4f}")
        logging.info(f"  Test ROC AUC: {roc_auc:.4f}")

        save_metrics(metrics_log_filename, args.epochs, "Test", test_loss, accuracy, balanced_accuracy, f1, roc_auc, flag=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Vision Transformer on CIFAR-100')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='learning rate for optimizer')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device to use for training (cuda/cpu)')
    parser.add_argument('--model-type', type=str, default='vanilla', help='variant to run')
    parser.add_argument('--n-blocks', type=int, default=4, help='number of transformer blocks')
    parser.add_argument('--d-hidden', type=int, default=64, help='hidden dimension of transformer block')
    parser.add_argument('--n-heads', type=int, default=2, help='number of attention heads')
    parser.add_argument('--log-dir', type=str, default='logs', help='directory to store logs')
    args = parser.parse_args()

    # CIFAR-100 Transformations
    cifar_train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    cifar_test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
    ])

    # Load CIFAR-100 Dataset
    train_dataset = CIFAR100(root='./cifar100', train=True, download=True, transform=cifar_train_transforms)
    test_dataset = CIFAR100(root='./cifar100', train=False, download=True, transform=cifar_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    main(train_loader=train_loader, test_loader=test_loader, args=args)
