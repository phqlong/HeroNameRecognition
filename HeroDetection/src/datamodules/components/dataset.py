import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class RandomTransform:
    def __init__(self, p=0.5, max_padding=100, noise=0.1):
        self.p = p
        self.max_padding = max_padding
        self.noise = noise
        
    def __call__(self, x):
        if random.random() < self.p:
            # Crop, Rotate, ColorJitter
            x = TF.rotate(x, angle=random.randint(-45, 45))
            x = TF.resized_crop(x, top=random.randint(0, 50), left=random.randint(0, 50), height=256, width=256, size=(256, 256))

            # Random padding
            padding = random.randint(0, self.max_padding)
            x = TF.pad(x, padding, padding_mode='reflect')
            
            # Pad with random colors
            padding_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            x = TF.pad(x, padding, padding_mode='constant', fill=padding_color)
            
            # Adjust brightness, contrast, and saturation
            brightness = random.uniform(0.7, 1.2)
            contrast = random.uniform(0.7, 1.2)
            saturation = random.uniform(0.7, 1.2)
            x = TF.adjust_brightness(x, brightness_factor=brightness)
            x = TF.adjust_contrast(x, contrast_factor=contrast)
            x = TF.adjust_saturation(x, saturation_factor=saturation)
            
            # Add blender nois
            x = TF.gaussian_blur(x, kernel_size=5)
            img_blended = Image.new(x.mode, x.size)
            x = Image.blend(x, img_blended, alpha=random.uniform(0.1, 0.5))
                                
        return x
    

class HeroImagesDataset(Dataset):
    def __init__(self, 
                 hero_images_path_list, 
                 hero_images_path_2_label,
                 num_triplets=20,
                 train=True,
                 to_tensor=True):
        
        self.hero_images_path_list = hero_images_path_list
        self.hero_images_path_2_label = hero_images_path_2_label
        self.num_triplets = num_triplets
        self.train = train
        self.to_tensor = to_tensor

        self.transform = transforms.Compose([
            transforms.Resize((40, 40)),
            transforms.Resize((256, 256)),
            RandomTransform(p=0.8),
            transforms.Resize((256, 256)),
        ])

        if self.to_tensor:
            self.transform_to_tensor = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),
            ])

        self.data, self.target = self.load_data()

    def load_data(self):
        data_samples = []
        targets = []

        for anchor_img_path in tqdm(self.hero_images_path_list):
            hero_label = self.hero_images_path_2_label[anchor_img_path]

            triplet_samples = self.create_triplet_samples(anchor_img_path)

            data_samples += triplet_samples
            targets += [hero_label] * len(triplet_samples)

        return data_samples, targets

    def create_triplet_samples(self, anchor_img_path):
        triplet_samples = []

        neg_img_paths = random.sample([p for p in self.hero_images_path_list if p != anchor_img_path], self.num_triplets)
        
        for neg_img_path in neg_img_paths:
            anchor_img = Image.open(anchor_img_path).convert('RGB')
            pos_img = Image.open(anchor_img_path).convert('RGB')
            neg_img = Image.open(neg_img_path).convert('RGB')

            # Apply transformations for positive and negative images
            pos_img = self.transform(pos_img)
            neg_img = self.transform(neg_img)

            # Apply transformations to tensor
            if self.to_tensor:
                anchor_img = self.transform_to_tensor(anchor_img)
                pos_img = self.transform_to_tensor(pos_img)
                neg_img = self.transform_to_tensor(neg_img)

            triplet_samples.append((anchor_img, pos_img, neg_img))

        return triplet_samples
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    
    def __len__(self):
        return len(self.data)