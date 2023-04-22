import os
import cv2
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image

from utils.utils import save_checkpoint, load_checkpoint, print_examples
from dataset import get_loader
from model import Transformer


def validation(model, valid_loader, valid_dataset, criterion, device):
    model.eval()

    total_loss = 0

    loader = tqdm(valid_loader, total=len(valid_loader), leave=True)
    with torch.no_grad():
        for idx, (images, captions) in enumerate(loader):
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions[:-1])
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[1:].reshape(-1))

            total_loss += loss.item()

            pred_captions = model.caption_image(images, valid_dataset.vocab)
            # result = ""
            # for i in pred_captions:
            #     result += i + " "

            images = ((images[0].permute(1, 2, 0).cpu().numpy() + 1) / 2) * 255
            # images = cv2.copyMakeBorder(images, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value=0)
            #
            # font = cv2.FONT_HERSHEY_SIMPLEX

            # cv2.putText(images, result, (128, 10), font, 1, (255, 255, 255), 2)
            images = cv2.cvtColor(images, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"results/images/img_{idx}.png", images)

            with open("results/res.txt", "a") as file:
                file.write(f"img_{idx}.png:   {pred_captions}\n")

            loader.set_postfix(avg_loss=total_loss / (idx + 1))

    return total_loss / len(valid_loader)


def train():
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_model = False
    save_model = True

    batch_size = 32

    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="datasets/MSCOCO2017/train",
        annotation_file="datasets/MSCOCO2017/annotations/captions_train2017.txt",
        transform=transform, num_workers=4, batch_size=batch_size
    )

    valid_loader, _ = get_loader(
        root_folder="datasets/MSCOCO2017/val",
        annotation_file="datasets/MSCOCO2017/annotations/captions_val2017.txt",
        transform=valid_transform, num_workers=4, batch_size=1, shuffle=False
    )

    embed_size = 1024
    patch_size = 16

    vocab_size = len(dataset.vocab)
    num_layers = 3
    learning_rate = 1e-4
    num_epochs = 100

    pad_idx = dataset.vocab.stoi["<PAD>"]

    model = Transformer(
        embed_size, vocab_size, num_layers, pad_idx, patch_size, device=device, heads=8, dim_head=128, drop=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if load_model:
        load_checkpoint(torch.load(f"e{embed_size}p{patch_size}n{num_layers}.pth.tar"), model, optimizer)

    count = 0
    min_train_loss = float("inf")
    min_valid_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        loader = tqdm(train_loader, total=len(train_loader), leave=True)
        for idx, (images, captions) in enumerate(loader):
            images = images.to(device)
            captions = captions.to(device)

            outputs = model(images, captions[:-1])

            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[1:].reshape(-1))

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

            total_loss += loss.item()
            loader.set_postfix(epoch=epoch, avg_loss=total_loss / (idx + 1))

        avg_valid_loss = validation(model, valid_loader, dataset, criterion, device)
        print_examples(model, device, dataset)

        if min_train_loss > total_loss / len(train_loader) or min_valid_loss > avg_valid_loss:
            min_train_loss = total_loss / len(train_loader)
            min_valid_loss = avg_valid_loss

            if save_model:
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()
                }
                save_checkpoint(checkpoint, f"e{embed_size}p{patch_size}n{num_layers}")

        else:
            count += 1

            if count == 7:
                break


if __name__ == "__main__":
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("results/images"):
        os.mkdir("results/images")

    train()
