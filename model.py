import torch
from torch import nn

from utils.TransformerEncoder import TransformerEncoder
from utils.TransformerDecoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers, pad_idx, patch_size,
                 forward_expansion=4, max_length=100, img_channels=3, heads=4,
                 dim_head=64, mlp_ratio=4, drop=0., device=torch.device("cuda")):
        super().__init__()
        self.device = device
        self.encoder = TransformerEncoder(
            patch_size, img_channels=img_channels,
            dim=embed_size, depth=num_layers, heads=heads,
            dim_head=dim_head, mlp_ratio=mlp_ratio,
            drop=drop
        ).to(device)

        self.decoder = TransformerDecoder(
            vocab_size, pad_idx, embed_size=embed_size, num_layers=num_layers, heads=heads,
            forward_expansion=forward_expansion, max_length=max_length
        ).to(device)

        self.num_layers = num_layers

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=100):
        sos_idx = vocabulary.stoi["<SOS>"]
        eos_idx = vocabulary.stoi["<EOS>"]

        with torch.no_grad():
            features = self.encoder(image).to(self.device)
            target_indices = [sos_idx]

            for i in range(max_length):
                captions = torch.LongTensor(target_indices).unsqueeze(1).to(self.device)
                output = self.decoder(features, captions)

                predicted = output.argmax(2)[-1, :].item()
                target_indices.append(predicted)

                if predicted == eos_idx:
                    break

            # print(target_indices.shape)
            target_tokens = [vocabulary.itos[i] for i in target_indices]
            return target_tokens


def testing():
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="datasets/flickr8k/images",
        annotation_file="datasets/flickr8k/captions.txt",
        transform=transform, num_workers=2, batch_size=1
    )

    pad_idx = dataset.vocab.stoi["<PAD>"]

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embed_size = 256
    patch_size = 64

    vocab_size = len(dataset.vocab)
    num_layers = 1
    num_epochs = 5

    model = Transformer(embed_size, vocab_size, num_layers, pad_idx, patch_size, device=device).to(device)
    encoder = TransformerEncoder(patch_size, dim=embed_size).to(device)
    decoder = TransformerDecoder(vocab_size, pad_idx, embed_size, num_layers).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    evaluate = True
    if evaluate:
        model.eval()

        # loader = tqdm(train_loader, total=len(train_loader), leave=True)
        for idx, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            # captions = captions.to(device)

            # print("\ncaptions main", captions.shape)

            pred_captions = model.caption_image(images, dataset.vocab)
            print("\n\npredicted captions shape", len(pred_captions))
            print(pred_captions, "\n")

    else:
        model.train()

        for epoch in range(num_epochs):
            # loader = tqdm(train_loader, total=len(train_loader), leave=True)
            for idx, (images, captions) in enumerate(train_loader):
                images = images.to(device)
                captions = captions.to(device)

                with torch.no_grad():
                    features = encoder(images)
                    outputs = decoder(features, captions[:-1])
                    # outputs = model(images, captions[:-1])

                    print("loss", criterion(outputs.reshape(-1, outputs.shape[2]), captions[1:].reshape(-1)))

            print_examples(model, device, dataset)


if __name__ == "__main__":
    from tqdm import tqdm
    import torchvision.transforms as transforms

    from utils.utils import print_examples
    from dataset import get_loader

    testing()
