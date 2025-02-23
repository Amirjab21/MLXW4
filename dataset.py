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
        all_captions = dataset['caption']
        number_of_captions = len(all_captions)

        tokenized = self.tokenizer(
            all_captions,
            padding='max_length',
            truncation=True,
            max_length=45,
            return_tensors="pt",
            pad_to_max_length=True  # Ensure consistent padding
            )
        tokenized_captions = tokenized.input_ids
        # random_caption_idx = torch.randint(0, len(captions), (1,)).item()
        # selected_caption = captions[random_caption_idx]

        # tokenized_caption = self.tokenizer(selected_caption, return_tensors="pt", padding="max_length", max_length=45, truncation=True)
        # input_ids = tokenized_captions['input_ids']
        eos_positions = (tokenized_captions == self.tokenizer.eos_token_id).nonzero()
        # Group EOS positions by caption (row)
        for i in range(len(all_captions)):
            caption_eos = eos_positions[eos_positions[:, 0] == i]
            if len(caption_eos) > 0:
                first_eos_pos = caption_eos[0][1]
                # Replace all tokens after the first EOS with PAD token (49408)
                tokenized_captions[i, first_eos_pos+1:] = 49408


        
    
        return {
            'images': image,
            'captions': tokenized_captions,
        }
    