import torch
import os
from PIL import Image
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocess import ImagePairDataset
from rrdb_arch import SRN

def main():
    train_dataset = ImagePairDataset("dataset/train")
    test_dataset = ImagePairDataset("dataset/val")

    print("dataset created!")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=6, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("data loaded!")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRN(1, 64, 512).to(device)

    print("model and device initialized!")
    print(f"device is: {device}!")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    print("initialized loss and optimizer!")

    #training loooop

    epochs = 10

    for i in range(epochs):
        model.train()
        total_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            print("done!")

            if i == 9:
                image = Image.fromarray(images)
                image.save(format='PNG')
            

        avg_loss = total_loss / len(train_dataset)
        print(f"Average loss for epoch {i + 1} was {avg_loss}.")

if __name__ == "__main__":
    main()