import torch
from transformers import BertTokenizer, CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np


processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def split_image_to_patches(image, image_height, image_width, patch_size, channels = 3):
    """
    Splits an image into patches, R first, then G, then B
    """
    blocks = []
    num_patches = image_height // patch_size
    image = np.array(image)  # This will give you a (3, 224, 224) numpy array
    print(image.shape)
    for c in range(channels):
        for i in range(num_patches):
            for j in range(num_patches):
                #left to right, top to bottom
                block = image[c, i*patch_size : (i+1)*patch_size , j*patch_size : (j+1)*patch_size]
                blocks.append(block)
    return blocks
    
    

class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dataset = self.dataset[idx]
        image = dataset['image_processed'] #height, width, channels
        captions = dataset['caption']
        
        image_processed = processor(images=np.array(image), return_tensors="pt", padding=True)
        image_processed = image_processed['pixel_values'].squeeze(0)
        # patches_array = split_image_to_patches(image, image.shape[0], image.shape[1], 32, 3)
        # patches_tensor = [torch.tensor(patch.flatten(), dtype=torch.float32) for patch in patches_array] 
        # image_tensor = torch.stack(patches_tensor)


        
        random_caption_idx = torch.randint(0, len(captions), (1,)).item()
        selected_caption = captions[random_caption_idx]
        tokenized_caption = self.tokenizer(selected_caption, return_tensors="pt", padding="max_length", max_length=30, truncation=True)

        
        
        return {
            'image': image_processed,
            'caption': tokenized_caption['input_ids'],
        }
    