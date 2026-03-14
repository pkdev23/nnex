import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from network import SmallNN


def get_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test,  batch_size=256, shuffle=False))


def train(epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SmallNN().to(device)
    opt    = optim.Adam(model.parameters(), lr=1e-3)

    train_loader, test_loader = get_loaders()

    print(f"🔧 Training auf {device}...")
    for epoch in range(1, epochs + 1):
        model.train()
        correct = 0
        total   = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad()
            out  = model(images)
            loss = F.cross_entropy(out, labels)
            loss.backward()
            opt.step()
            correct += (out.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        print(f"  Epoche {epoch} → Train Acc: {100*correct/total:.1f}%")

    # Test Accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out      = model(images)
            correct += (out.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    print(f"\n✅ Test Accuracy: {100*correct/total:.1f}%")

    # Modell speichern
    torch.save(model.state_dict(), "model.pth")
    print("💾 Modell gespeichert: model.pth")
    return model


if __name__ == "__main__":
    train()