import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        for label, cls in enumerate(["real", "fake"]):
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append(
                        (os.path.join(cls_path, file), label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, label
