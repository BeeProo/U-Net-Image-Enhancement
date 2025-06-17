import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt

# ==================== UNET MODEL ====================

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = CBR(3, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.middle = CBR(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        mid = self.middle(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(mid), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

# ==================== DATASET ====================

class ImageEnhancementDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.images = sorted(os.listdir(input_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.images[idx])
        target_path = os.path.join(self.target_dir, self.images[idx])
        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(target_path).convert("RGB")

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

# ==================== MAIN ====================

def main():
    # ----- Setup -----
    input_folder = "dataset/inputs"
    target_folder = "dataset/targets"
    save_model_path = "unet_image_enhancer.pth"

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])

    dataset = ImageEnhancementDataset(input_folder, target_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # ----- Training -----
    print("🚀 Starting training...")
    print(f"Found {len(dataloader)} batches in training dataset.")
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

    # Save model
    torch.save(model.state_dict(), save_model_path)
    print(f"Done! Model saved as: {save_model_path}")

    # ----- Test on One Image -----
    test_image_path = "test_image.jpg"

    if os.path.exists(test_image_path):
        model.eval()
        image = Image.open(test_image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor).squeeze().cpu().clamp(0, 1)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Input")
        plt.imshow(image.resize((256, 256)))
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Enhanced Output")
        plt.imshow(output.permute(1, 2, 0))
        plt.axis("off")
        plt.show()
    else:
        print("oh no! test image not found — skipping visualization.")

if __name__ == "__main__":
    main()
