from datetime import datetime
from pathlib import Path

import torch
from utils import load_data
from models import FaceDiseaseCNN, ClassficationLoss
from train import train, evaluate
from pathlib import Path

import torch.utils.tensorboard as tb
from models import FaceDiseaseCNN, ClassficationLoss



def main(epochs=50, batch_size=64, learning_rate=0.0001):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader = load_data(split="train", batch_size=batch_size, image_size=224, shuffle=True)
    val_loader = load_data(split="testing", batch_size=batch_size, image_size=224, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = FaceDiseaseCNN().to(device)
    print(model)
    loss_fn = ClassficationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # TensorBoard setup
    log_dir = Path("logs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = tb.SummaryWriter(log_dir)
    
    global_step = 0
    num_epochs = epochs
    for epoch in range(num_epochs):
        train_accuracy, global_step = train(model, train_loader, loss_fn, device, optimizer, global_step, writer)
        val_accuracy = evaluate(model, val_loader, loss_fn, device, global_step, writer)
        
        epoch_train_acc = torch.as_tensor(train_accuracy).mean().item()
        epoch_val_acc = torch.as_tensor(val_accuracy).mean().item()

        writer.add_scalar("Accuracy/Train", epoch_train_acc, global_step)
        writer.add_scalar("Accuracy/Validation", epoch_val_acc, global_step)
        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epochs - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epochs:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

def analyze_data():
    from utils import FaceDiseaseDataset, LABEL_NAMES
    dataset = FaceDiseaseDataset(Path("data/train"))
    class_counts = {label: 0 for label in LABEL_NAMES}
    for _, label in dataset:
        class_counts[LABEL_NAMES[label]] += 1
    print("Dataset class distribution:")
    for name, count in class_counts.items():
        print(f"  {name}: {count} images ({count / len(dataset) * 100:.1f}%)")

if __name__ == "__main__":
    analyze_data()
    #main()