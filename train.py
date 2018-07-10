import argparse
import os
import pickle

import torch
import torch.nn as nn

import dataloader
import mappings
import model



def main(args):
    device = torch.device(args.device)
    # Model will be saved into directory at every epoch
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    with open(args.vocab_path,"rb") as f:
        vocab = pickle.load(f)
    with open(args.ans_path,"rb") as f:
        answers = pickle.load(f)
    
    # Dataloader
    data_loader = dataloader.get_loader(args.image_dir,
                                        args.annotation_path,
                                        args.question_path,
                                        vocab, answers,
                                        batch_size=2, shuffle=False
                                       )
    
    # Model
    # Note: Expected values: len(vocab) = 8254, len(answers) = 3001
    net = model.VQAnet(len(vocab), embedding_size=128, lstm_size=512,
                       fc_size=1024, num_answers=len(answers))
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # Note: parameters of optimizer are actually defaults.
    optimizer = torch.optim.Adam(net.trainable_parameters(), lr=0.001, betas=(0.9, 0.999))
    # Will likely never reach 12 epochs, but appoximately follows what paper uses
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 12, 0.5)
    
    iterations_per_epoch = len(data_loader)
    for epoch in range(args.num_epochs):
        for i, (images, qn_idxs, ans_idxs) in data_loader:
            images.to(device)
            if device.type=="cuda":
                qn_idxs.cuda()
            elif device.type=="cpu":
                qn_idxs.cpu()
            ans_idxs.to(device) # May not be necessary
            
            
    
    # DEBUG
    """
    st = torch.load(os.path.join(args.model_path,"net.pth"))
    net.load_state_dict(st)
    for param in net.parameters():
        print(param.requires_grad)
    #torch.save(net.state_dict(), os.path.join(args.model_path,"net.pth"))
    """



if __name__ == '__main__' or True: # DEBUG
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--ans_path', type=str, default='answers.pkl', help='path for answers wrapper')
    parser.add_argument('--image_dir', type=str, default='images/train2014', help='directory for images')
    parser.add_argument('--annotation_path', type=str,
                        default='annotations/v2_mscoco_train2014_annotations.json',
                        help='path for annotations json')
    parser.add_argument('--question_path', type=str,
                        default='questions/v2_OpenEnded_mscoco_train2014_questions.json',
                        help='path for questions json')
    parser.add_argument('--device', type=str, default="auto", choices=("auto","cpu","cuda"), help='device to use')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to train for')
    
    args = parser.parse_args()
    
    if args.device is "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(args.device), flush=True)
    
    net = main(args)
    """ For # DEBUG
    with open(args.vocab_path,"rb") as f:
        vocab = pickle.load(f)
    with open(args.ans_path,"rb") as f:
        answers = pickle.load(f)
    
    data_loader = dataloader.get_loader(args.image_dir,
                                        args.annotation_path,
                                        args.question_path,
                                        vocab, answers,
                                        2, False
                                       )
                                       """