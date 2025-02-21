import torch
from transformers import BertTokenizer, CLIPModel, CLIPProcessor
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from torchvision import transforms

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
])
    

class Flickr30kDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        tokenizer.add_special_tokens({"pad_token": "<<<PAD>>>"})

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        dataset = self.dataset[idx]
        image = transform(dataset['image']) #height, width, channels
        captions = dataset['caption']
        random_caption_idx = torch.randint(0, len(captions), (1,)).item()
        selected_caption = captions[random_caption_idx]

        tokenized_caption = self.tokenizer(selected_caption, return_tensors="pt", padding="max_length", max_length=45, truncation=True)
        input_ids = tokenized_caption['input_ids']

        eos_positions = (input_ids == self.tokenizer.eos_token_id).nonzero()
        if len(eos_positions) > 0:
            first_eos_pos = eos_positions[0][1]
            
            # Replace all padding tokens after the first EOS with 49408 (<<<PAD>>>)
            # Keep one EOS token (49407) at the first EOS position
            input_ids[0, first_eos_pos+1:] = 49408

        tokenized_caption['input_ids'] = input_ids


        

    
        return {
            'image': image,
            'caption': tokenized_caption['input_ids'].squeeze(0),
        }
    