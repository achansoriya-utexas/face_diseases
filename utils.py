import random
import shutil
from collections import Counter
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

LABEL_NAMES = ['Acne', 'Actinic Keratosis', 'Basal Cell Carcinoma', 'Eczemaa', 'Rosacea']

DATA_DIR = Path(__file__).parent / "data"


class FaceDiseaseDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for label_idx, label_name in enumerate(LABEL_NAMES):
            label_dir = root / label_name
            if not label_dir.exists():
                continue
            for img_path in sorted(label_dir.iterdir()):
                if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp'):
                    self.samples.append((img_path, label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _download_dataset() -> Path:
    """Download dataset from Kaggle and copy to local data/ folder if needed."""
    if DATA_DIR.exists() and (DATA_DIR / "train").exists():
        return DATA_DIR
    cache_path = Path(kagglehub.dataset_download("amellia/face-skin-disease"))
    src = cache_path / "DATA"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "testing"):
        dst = DATA_DIR / split
        if not dst.exists():
            shutil.copytree(src / split, dst)
    return DATA_DIR


def load_data(
    split: str = "train",
    batch_size: int = 32,
    image_size: int = 224,
    shuffle: bool | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Download (if needed) and return a DataLoader for the given split.

    Args:
        split: "train" or "testing".
        batch_size: Batch size.
        image_size: Resize target (square).
        shuffle: Shuffle the data. Defaults to True for train, False for testing.
        num_workers: DataLoader workers.
    """
    data_root = _download_dataset()

    if shuffle is None:
        shuffle = split == "train"

    if split == "train":
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    dataset = FaceDiseaseDataset(data_root / split, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def analyze_data():
    """Analyze the training dataset and display plots with matplotlib."""
    data_root = _download_dataset()
    dataset = FaceDiseaseDataset(data_root / "train")

    # Count samples per class
    label_counts = Counter(label for _, label in dataset.samples)
    class_names = [LABEL_NAMES[i] for i in range(len(LABEL_NAMES))]
    counts = [label_counts.get(i, 0) for i in range(len(LABEL_NAMES))]

    # Collect image sizes
    widths, heights = [], []
    for img_path, _ in dataset.samples:
        with Image.open(img_path) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Face Disease Training Data Analysis", fontsize=16, fontweight="bold")

    # 1. Class distribution bar chart
    ax = axes[0, 0]
    bars = ax.bar(class_names, counts, color=plt.cm.Set2.colors[:len(class_names)])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Number of Images")
    ax.tick_params(axis="x", rotation=30)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha="center", fontweight="bold")

    # 2. Class distribution pie chart
    ax = axes[0, 1]
    ax.pie(counts, labels=class_names, autopct="%1.1f%%",
           colors=plt.cm.Set2.colors[:len(class_names)], startangle=90)
    ax.set_title("Class Proportions")

    # 3. Image resolution scatter plot
    ax = axes[1, 0]
    labels_arr = np.array([label for _, label in dataset.samples])
    for i, name in enumerate(class_names):
        mask = labels_arr == i
        ax.scatter(np.array(widths)[mask], np.array(heights)[mask],
                   label=name, alpha=0.6, s=20)
    ax.set_title("Image Resolutions")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Height (px)")
    ax.legend(fontsize=8)

    # 4. Sample images grid (one random image per class)
    ax = axes[1, 1]
    ax.set_title("Sample Images (1 per class)")
    ax.axis("off")
    samples_per_class = {}
    for img_path, label in dataset.samples:
        if label not in samples_per_class:
            samples_per_class[label] = []
        samples_per_class[label].append(img_path)

    grid_axes = ax.inset_axes([0, 0, 1, 1])
    grid_axes.axis("off")
    n_classes = len(class_names)
    for i in range(n_classes):
        inset = ax.inset_axes([i / n_classes, 0.0, 1 / n_classes, 0.75])
        img_path = random.choice(samples_per_class[i])
        img = Image.open(img_path).convert("RGB").resize((112, 112))
        inset.imshow(img)
        inset.set_title(class_names[i], fontsize=7)
        inset.axis("off")

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"\nTotal training samples: {len(dataset)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Image size range: {min(widths)}x{min(heights)} - {max(widths)}x{max(heights)}")
    for name, count in zip(class_names, counts):
        print(f"  {name}: {count} images ({count / len(dataset) * 100:.1f}%)")


def compute_accuracy(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute accuracy given predicted logits and true labels."""
    output_indexes = preds.max(1)[1].type_as(labels)
    return (output_indexes == labels).float().mean()
