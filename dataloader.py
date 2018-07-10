import os

import nltk
import torch
import torch.utils.data as data
import torch.nn.utils.rnn as rnn
import torchvision.transforms as transforms
from PIL import Image

from vqaTools.vqa import VQA



class VqaDataset(data.Dataset):
    """VQA Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, annotation_file, question_file, vocab, answers, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            annotation_file: path to vqa annotation file
            question_file: path to vqa question file
            vocab: vocabulary wrapper.
            answer: answer wrapper
            transform: image transformer.
        """
        self.root = root
        self.vqa = VQA(annotation_file, question_file)
        self.image_format = "COCO_{}_{{:012}}.jpg".format(self.vqa.dataset['data_subtype'])
        self.qids = [ann["question_id"] for ann in self.vqa.dataset["annotations"]]
        self.vocab = vocab
        self.answers = answers
        self.transform = transform
    
    def load_image(self, image_id):
        filename = self.image_format.format(image_id)
        path = os.path.join(self.root, filename)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        return img
    
    def __getitem__(self, index):
        """Returns one data element,
        (image, question as tensor, answers as tensor)."""
        vqa = self.vqa
        vocab = self.vocab
        answers = self.answers
        
        qid = self.qids[index]
        ann = vqa.qa[qid]
        question = vqa.qqa[qid]
        
        # Get image
        image = self.load_image(question["image_id"])
        if self.transform is not None:
            image = self.transform(image)
        
        # Get question
        qn = question["question"]
        tokens = nltk.tokenize.word_tokenize(str(qn).lower())
        qn_idx = []
        qn_idx.append(vocab('<start>'))
        qn_idx.extend(vocab(token) for token in tokens)
        qn_idx.append(vocab('<end>'))
        qn_idx = torch.tensor(qn_idx, dtype=torch.long)
        
        # Get answers
        ans_gen = (a["answer"] for a in ann["answers"])
        ans_idx = [answers(a) for a in ans_gen]
        ans_idx = torch.tensor(ans_idx, dtype=torch.long)
        
        return image, qn_idx, ans_idx

    def __len__(self):
        return len(self.qids)



def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    
    Args:
        data: list of tuple (image, qn_idx, ans_idx). 
            - image: torch tensor of shape (3, height, width).
                     (height and witdh should be constant for all images,
                      transform if necessary)
            - qn_idx: torch tensor of shape (question_length).
            - ans_idx: torch tensor of shape (number_of_answers).
                       (There should always be 10/constant number of answers)
    
    Returns:
        images: torch tensor of shape (batch_size, 3, height, width).
        qn_idxs: torch packed sequence for questions in batch.
        ans_idxs: torch tensor of shape (batch_size, number_of_answers).
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, qn_idxs, ans_idxs = zip(*data)
    
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)
    
    # Merge questions (from tuple of 1D tensor to PackedSequence).
    qn_idxs = rnn.pack_sequence(qn_idxs)
    
    # Merge answers (from tuple of 1D tensor to 2D tensor).
    ans_idxs = torch.stack(ans_idxs, 0)
    
    return images, qn_idxs, ans_idxs

def get_loader(root, annotation_file, question_file, vocab, answers,
               batch_size, shuffle, num_workers=0, transform=None):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    # vqa dataset
    vqa = VqaDataset(root=root,
                     annotation_file=annotation_file,
                     question_file=question_file,
                     vocab=vocab,
                     answers=answers,
                     transform=transform)
    
    # Data loader for dataset
    # This will return (images, qn_idxs, ans_idxs) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224). (or batch_size,height,width)
    # qn_idxs: a PackedSequence.
    # ans_idxs: a tensor of shape (batch_size, 10). (or batch_size, number_of_answers)
    data_loader = torch.utils.data.DataLoader(dataset=vqa, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader