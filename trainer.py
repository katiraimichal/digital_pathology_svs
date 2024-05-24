import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc, classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, accuracy_score, f1_score
import seaborn as sns


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs, device, save_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.device = device
        self.save_dir = save_dir
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train(self):
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
                loss = self.criterion(outputs, one_hot_labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)
                predicted_labels = torch.argmax(outputs, dim=1)
                correct += (predicted_labels.cpu() == labels.cpu()).sum().item()
                total += labels.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            epoch_accuracy = correct / total
            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_accuracy)
            print(
                f'Train - Epoch [{epoch + 1}/{self.num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

            self.validate(epoch)

        self.plot_metrics()

    def validate(self, epoch):
        self.model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
                loss = self.criterion(outputs, one_hot_labels)
                val_running_loss += loss.item() * images.size(0)
                predicted_labels = torch.argmax(outputs, dim=1)
                correct += (predicted_labels.cpu() == labels.cpu()).sum().item()
                total += labels.size(0)

        val_epoch_loss = val_running_loss / len(self.val_loader.dataset)
        val_accuracy = correct / total
        self.val_losses.append(val_epoch_loss)
        self.val_accuracies.append(val_accuracy)
        print(
            f'Validation - Epoch [{epoch + 1}/{self.num_epochs}], Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.4f}')

    def test(self):
        self.model.eval()
        test_correct = 0
        test_total = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                predicted_labels = torch.argmax(outputs, dim=1)
                test_correct += (predicted_labels.cpu() == labels.cpu()).sum().item()
                test_total += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted_labels.cpu().numpy())

        test_accuracy = test_correct / test_total
        print(f'Test Accuracy: {test_accuracy:.4f}')

        # Calculate and plot ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(self.save_dir, 'roc_curve.png'))
        plt.close()

        # Calculate and plot Precision-Recall curve
        precision, recall, _ = precision_recall_curve(all_labels, all_preds)

        plt.figure()
        plt.plot(recall, precision, color='darkorange', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(os.path.join(self.save_dir, 'pr_curve.png'))
        plt.close()

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
        plt.close()

        # Additional Statistics:
        test_accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds)

        with open(os.path.join(self.save_dir, 'test_metrics.txt'), 'w') as f:
            f.write(f'Test Accuracy: {test_accuracy:.4f}\n')
            f.write(f'Test F1 Score: {f1:.4f}\n')
            f.write(report)

    def plot_metrics(self):
        epochs = range(1, self.num_epochs + 1)

        # Plot loss
        plt.figure()
        plt.plot(epochs, self.train_losses, label='Train Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.save_dir, 'loss_plot.png'))
        plt.close()

        # Plot accuracy
        plt.figure()
        plt.plot(epochs, self.train_accuracies, label='Train Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Over Epochs')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.save_dir, 'accuracy_plot.png'))
        plt.close()
