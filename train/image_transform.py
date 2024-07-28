import torchvision.transforms as transforms

IMAGE_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=[448, 448])
])