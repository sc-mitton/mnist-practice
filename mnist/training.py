import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class Training:

    def __init__(self,
                 model,
                 val_loader,
                 train_loader,
                 optimizer=optim.Adam,
                 lr=.001):

        self.model = model
        self.optimizer_type = optimizer
        self.lr = lr
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.loader = None
        self.optimizer = self.optimizer_type(
            params=self.model.parameters(), lr=self.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.5)

    def train(self, epochs=2):
        self.model.train()

        lossi = []
        val_lossi = []

        for e in range(epochs):
            self.loader = self.train_loader
            print('epoch: ', e, 'training on: ', len(self.loader), 'examples')
            total_loss = 0.0
            num_batches = 0

            for x, target in self.loader:
                output = self.model(x)
                loss = F.cross_entropy(output, target)

                self.optimizer.zero_grad()
                loss.backward()

                total_loss += loss.item()
                num_batches += 1

                self.optimizer.step()

            if e % 5 == 0:
                self.scheduler.step()

            lossi.append(total_loss / num_batches)
            val_lossi.append(self.validate())

        return lossi, val_lossi

    def validate(self):
        return self.test(self.val_loader)

    def test(self, loader=None):
        self.model.eval()

        self.loader = loader

        total_loss = 0.0
        num_batches = 0

        correct = 0  # Correct predictions counter
        total = 0  # Total number of images

        print('validating on ', len(self.loader), 'examples')
        with torch.no_grad():
            for x, target in iter(self.loader):
                output = self.model(x)
                loss = F.cross_entropy(output, target)

                predictions = output.argmax(dim=1)

                total_loss += loss.item()
                correct += (predictions == target).sum().item()
                total += target.size(0)
                num_batches += 1

        avg_loss = total_loss / num_batches
        print('correct: ', correct, 'total: ', total)
        accuracy = (correct / total) * 100  # Accuracy in percentage

        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

        return avg_loss
