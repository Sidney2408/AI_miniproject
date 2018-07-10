import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn

import dataloader
import mappings
import model

def unroll_ans(ans_idxs):
    ans_idxs = ans_idxs.numpy()
    # Unique answers for each question
    ans_unique = []
    # weights of each unique answer for each question (in order of ans_unique)
    # weight is proportion of frequency of that answer to all non-<don't know> answers
    ans_weights = []
    # Number of unique answers for each question,
    # equals number of times result has to be repeated to match dimensions
    ans_repeats = []
    for row in ans_idxs:
        row = row[row!=0] # remove 0s, which correspond to <don't know>.
        num_nonzero = len(row)
        if num_nonzero == 0:
            ans_unique.append(np.array([], dtype=np.int64))
            ans_weights.append(np.array([], dtype=np.float32))
            ans_repeats.append(0)
            continue
        
        uniq, counts = np.unique(row, return_counts=True)
        weights = counts/float(num_nonzero)
        ans_unique.append(uniq)
        ans_weights.append(weights)
        ans_repeats.append(len(uniq))
    # Combine into tensors
    ans_unique = np.concatenate(ans_unique)
    ans_unique = torch.tensor(ans_unique)
    ans_weights = np.concatenate(ans_weights)
    ans_weights = torch.tensor(ans_weights)
    return ans_unique, ans_weights, ans_repeats

def repeat(tensor, times):
    if len(tensor) != len(times):
        raise ValueError("Length of tensor and times not equal! tensor:{} times:{}".format(len(tensor), len(times)))
    
    repeated = []
    for row, t in zip(tensor, times):
         # Prevent empty tensors from being appended.
         # (causes .backward() to fail)
        if t>0:
            repeated.append(row.expand(t,-1))
    repeated = torch.cat(repeated, dim=0)
    return repeated

def main(args):
    device = torch.device(args.device)
    # Model will be saved into directory at every epoch
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    # Load the latest model.
    # TODO
    
    with open(args.vocab_path,"rb") as f:
        vocab = pickle.load(f)
    with open(args.ans_path,"rb") as f:
        answers = pickle.load(f)
    
    # Dataloader
    data_loader = dataloader.get_loader(args.image_dir,
                                        args.annotation_path,
                                        args.question_path,
                                        vocab, answers,
                                        batch_size=8, shuffle=False # DEBUG
                                        #batch_size=args.batch_size, shuffle=True
                                       )
    
    # Model. Parameters are fixed.
    # Note: Expected values: len(vocab) = 8254, len(answers) = 3001
    net = model.VQAnet(len(vocab), embedding_size=128, lstm_size=512,
                       fc_size=1024, num_answers=len(answers))
    net.to(device)
    
    # Don't reduce loss, so can manually weight by answers.
    criterion = nn.CrossEntropyLoss(reduce=False)
    # Note: parameters of optimizer are actually defaults.
    optimizer = torch.optim.Adam(net.trainable_parameters(), lr=0.001, betas=(0.9, 0.999))
    # Will likely never reach 12 epochs, but appoximately follows what paper uses
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 12, 0.5)
    
    iterations_per_epoch = len(data_loader)
    for epoch in range(args.num_epochs):
        scheduler.step()
        epoch_start = time.time()
        total_loss = 0
        
        for i, (images, qn_idxs, ans_idxs) in enumerate(data_loader,1):
            images = images.to(device)
            if device.type=="cuda":
                qn_idxs = qn_idxs.cuda()
            elif device.type=="cpu":
                qn_idxs = qn_idxs.cpu()
            ans_unique, ans_weights, ans_repeats = unroll_ans(ans_idxs)
            ans_unique = ans_unique.to(device)
            ans_weights = ans_weights.type(torch.float).to(device)
            
            if sum(ans_repeats) != 0:
                net.zero_grad()
                
                output = net(images, qn_idxs)
                output = repeat(output, ans_repeats)
                losses = criterion(output, ans_unique)
                loss = (losses*ans_weights).sum() / ans_weights.sum()
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if i%100 == 0:
                current_time = time.time() - epoch_start
                current_avg_loss = total_loss/i
                print("Epoch {:2}, Step: {:6}/{:6}, Loss:{:.5f}, time:{:.3f}s".format(
                       epoch, i, iterations_per_epoch, current_avg_loss, current_time),
                       flush=True, end="\r"
                     )
        current_time = time.time() - epoch_start
        current_avg_loss = total_loss/i
        print("Epoch {:2}, Step: {:6}/{:6}, Loss:{:.5f}, time:{:.3f}s, EPOCH COMPLETE".format(
                       epoch, i, iterations_per_epoch, current_avg_loss, current_time),
                       flush=True
                     )
        
        # Save model
        # TODO



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
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    # Note: batch_size 8 seems to consume 1GB of GPU memory(?)
    
    args = parser.parse_args()
    
    if args.device is "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(args.device), flush=True)
    
    main(args)