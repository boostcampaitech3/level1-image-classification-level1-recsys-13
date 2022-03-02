from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize, Lambda, RandomHorizontalFlip, ToPILImage, CenterCrop, Grayscale

def get_transform(config):
    transform = {
        "main": transforms.Compose(
            [
                Resize(config.image_size),
                ToTensor(),
                Normalize(mean=config.image_normal_mean, std=config.image_normal_std),
                Grayscale(num_output_channels=3),
            ]
        ),
        "genderModel": transforms.Compose(
            [
                Resize(config.image_size),
                ToTensor(),
                Normalize(mean=config.image_normal_mean, std=config.image_normal_std),
                Grayscale(num_output_channels=3),
            ]
        ),
        "ageModel": transforms.Compose(
            [
                Resize(config.image_size),
                ToTensor(),
                Normalize(mean=config.image_normal_mean, std=config.image_normal_std),
                Grayscale(num_output_channels=3),
            ]
        ),

    }
    return transform