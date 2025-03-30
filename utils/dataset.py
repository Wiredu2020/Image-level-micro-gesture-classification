from pickle import FALSE
from .config import CFG
import os
import torch
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from collections import Counter
from sklearn.utils import resample
import cv2


label_encodings = {
                1:"Turtle neck",
                2: "Bulging face or deep breath",
                3: "Touch Hat",
                4: "Touching or scratching head",
                5: "Touching or scratching forehead", 
                6: "Cover face",
                7: "Rubbing eyes", 
                8: "Touching or Scratching facial parts", 
                9: "Touching ears", 
                10: "Biting nails", 
                11: "touching jaw",
                12: "Touching or scratching neck", 
                13: "Playing or adjusting hair", 
                14: "Buckle botton, pulling shirt collar, adjusting tie",
                15: "Touching or covering suprasternal notch", 
                16: "Scraching back",
                17: "Folding arms",
                18: "Dustoffing clothes",
                19: "Puttting arms behind body", 
                20: "Moving torso",
                21: "Sitting straightly",
                22: "scraching or touching arms",
                23: "Rubbing or holding hands",
                24: "Crossing fingers",
                25: "Minaret gesture",
                26: "Playing or manipulating objects",
                27: "Hold back arms",
                28: "Head up",
                29: "Pressing lips", 
                30: "Arms akimbo", 
                31: "Shaking shoulders",
                32: "Illustrative hand gestures" 
}

class MGDataset(Dataset):
    """
    Micro Gesture dataset builder
    """
    def __init__(self, root_dir, labels_enc = False , tokenizer= None, aug = CFG.data_aug ):
        """
        Initializes the dataset.

        Args:
            image_filenames (array): Array of image filenames.
            captions (array): Array of corresponding captions.
            tokenizer (DistilBertTokenizer): Tokenizer for encoding captions.
            transforms (albumentations.Compose): Image transformations.
        """
        self.root_dir = root_dir
        #self.image_filenames = image_filenames
        self.labels_enc = labels_enc
        if self.labels_enc:
            self.samples = []
            self.cap_data = self.labels_enc
            for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
                class_dir = os.path.join(root_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        cap = self.cap_data[class_idx+1]
                        self.samples.append((img_path, cap))
            

            if aug:
                class_samples = {}
                for img_path, label in self.samples:
                    class_samples.setdefault(label, []).append((img_path, label))
                
                # Find the minimum class count
                min_count = min(len(v) for v in class_samples.values())
                
                downsampled_samples = []
                for label, items in class_samples.items():
                    if len(items) > min_count:
                        # Downsample to match min_count
                        items_downsampled = resample(
                            items,
                            replace=False,
                            n_samples=min_count,
                            random_state=42
                        )
                        downsampled_samples.extend(items_downsampled)
                    else:
                        downsampled_samples.extend(items)
                
                random.shuffle(downsampled_samples)
                self.samples = downsampled_samples

            self.captions = [gesture[1] for gesture in self.samples]
            self.encoded_captions = tokenizer(
                self.captions, padding=True, truncation=True, max_length=CFG.max_length
            )
        else:
            self.samples = []
            for class_idx, class_name in enumerate(sorted(os.listdir(root_dir))):
                class_dir = os.path.join(root_dir, class_name)
                if os.path.isdir(class_dir):
                    for img_name in os.listdir(class_dir):
                        img_path = os.path.join(class_dir, img_name)
                        self.samples.append(img_path)
    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            dict: Dictionary containing image, caption, and encoded caption.
        """
        if self.labels_enc:
            item = {
                key: torch.tensor(values[idx])
                for key, values in self.encoded_captions.items()
            }
        else:
            item = {}
        img_path, label = self.samples[idx]
        image = cv2.imread(f"{img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = A.Compose(
            [
                A.Resize(CFG.size, CFG.size), #always_apply=True
                A.Normalize(max_pixel_value=255.0),#always_apply=True
                #A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ])
        image = transform(image=image)['image']
        #item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['image'] = image
        if self.labels_enc:
            item['caption'] = label

        return item


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)



def MGDataloader(dataset, data_set = "Train"):
    """
    Build data loaders for training/validation.

    Args:
        dataset: Takes the looaded dataset 
        data_set: Specifies the type for batch formation and shuffling
    Returns:
        torch.utils.data.DataLoader: Data loader instance.
    """
    dataloader = DataLoader(
        dataset,
        batch_size= 1 if data_set == "test" else CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if data_set == "train" else False,
    )
    return dataloader