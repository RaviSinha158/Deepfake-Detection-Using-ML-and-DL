import torch
from torch.utils.data import DataLoader
from dataset import DeepfakeDataset
from model import DeepfakeModel

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load test dataset
test_dataset = DeepfakeDataset("data/test")
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load trained model
model = DeepfakeModel().to(device)
model.load_state_dict(torch.load("models/deepfake_model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ Test Accuracy: {accuracy:.2f}%")
