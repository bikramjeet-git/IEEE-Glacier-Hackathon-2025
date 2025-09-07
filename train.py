import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import rasterio
from tqdm import tqdm

from logic import UNet  


class GlacierDataset(Dataset):
    def __init__(self, root_dir, bands=['B2','B3','B4','B6','B10'], transform=None):
        self.root_dir = Path(root_dir)
        self.bands = bands
        self.transform = transform

       
        band1_dir = self.root_dir / "Band1"
        self.ids = [f.name.split("_", 3)[-1].replace(".tif", "") for f in band1_dir.glob("*.tif")]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        image_data = []

        for band in self.bands:
            band_folder = f"Band{int(band[1])-1}" 
            band_file = list((self.root_dir / band_folder).glob(f"{band}_*_{image_id}.tif"))[0]

            with rasterio.open(band_file) as src:
                band_arr = src.read(1).astype(np.float32)
                image_data.append(band_arr)

        image_np = np.stack(image_data, axis=0) / 65535.0
        image_tensor = torch.from_numpy(image_np).float()

      
        mask_file = self.root_dir / "Mask" / f"mask_{image_id}.tif"
        with rasterio.open(mask_file) as src:
            mask = src.read(1).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0) 

        return image_tensor, mask_tensor


def train_model(train_loader, val_loader, device, epochs=20, lr=1e-3):
    model = UNet(n_channels=5, n_classes=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device).float()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

     
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device).float()
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f} .... Val Loss {avg_val_loss:.4f}")


        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "model.pth")
            print(f"modeel updated (Val Loss: {best_val_loss:.4f})")

    print("saved as model.pth")
    return model


if __name__ == "__main__":
    dataset = GlacierDataset("Train")
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_model(train_loader, val_loader, device, epochs=30, lr=1e-4)
