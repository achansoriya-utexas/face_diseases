
from torch.utils.data import DataLoader, Dataset
import tensorboard as tb
from models import FaceDiseaseCNN, ClassficationLoss



def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def main():
    cnn = FaceDiseaseCNN()
    print(cnn)
    print("Number of parameters:", sum(p.numel() for p in cnn.parameters()))


if __name__ == "__main__":
    main()
