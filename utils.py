import glob
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


class CircleCrop(object):
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):
        # Convert PIL image to PyTorch tensor
        img = transforms.ToTensor()(img)
        
        # Define circular mask
        mask = np.zeros((img.shape[-2], img.shape[-1]))
        center = [img.shape[-1] / 2, img.shape[-2] / 2]
        radius = min(img.shape[-1], img.shape[-2]) / 2
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if (i - center[1]) ** 2 + (j - center[0]) ** 2 <= radius ** 2:
                    mask[i, j] = 1
        
        # Apply mask to image tensor
        img = img * mask
        
        # Convert tensor back to PIL image
        img = transforms.ToPILImage()(img)
        
        # Apply additional transforms
        transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.CenterCrop(self.size)
        ])
        img = transform(img)
        
        return img


def get_embedding(model, image_path, is_test=True, downsize=40):
    if is_test:
        transform = transforms.Compose([
                        transforms.Lambda(lambda x: x.crop((0, 0, int(x.height*1.2), x.height))),  # Crop the left side
                        transforms.Resize((256, 256)),
                        transforms.CenterCrop(175),
                        # CircleCrop(size=200),  # Crop the left side
                        # transforms.Resize((256, 256)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]),
                    ])
    else:
        transform = transforms.Compose([
                        transforms.Resize((downsize, downsize)),
                        transforms.Resize((256, 256)),
                        CircleCrop(size=256),  # Crop the left side
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]),
                    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        embedding = model(image).squeeze().cpu().detach().numpy()
        return embedding
    

def get_hero_path_list(hero_images_dir):
    # Get hero names list
    # with open(HERO_NAMES_PATH, "r") as f:
    #     hero_names = f.read().splitlines()

    hero_images_path_list = glob.glob(hero_images_dir+"*")

    # Get labels for hero images
    hero_images_path_2_label = {path: path.split("\\")[-1].split(".")[0] for path in hero_images_path_list}

    return hero_images_path_list, hero_images_path_2_label


def get_test_path_list(test_images_dir, test_labels_path):
    test_images_path_list = glob.glob(test_images_dir+"*")

    # Get labels for test images
    with open(test_labels_path, "r") as f:
        test_labels = f.read().splitlines()
        test_file_2_labels = [label.split("\t") for label in test_labels]
        test_file_2_labels = {label[0]: label[1] for label in test_file_2_labels}

    return test_images_path_list, test_file_2_labels


def path_2_label(path, test_file_2_labels):
    file_name = path.split("\\")[-1]
    return test_file_2_labels[file_name]
