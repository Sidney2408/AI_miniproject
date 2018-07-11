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

def evaluate_accuracy(net_answers, gt_answers):
    """Compute list of accuracies of all questions in minibatch.
    
    net_answers is the classification weights.
    It is a tensor of shape (batch_size, net_num_answers)
    net_num_answers is the number of classes the net classifies (should be 3001).
    
    gt_answers is the ground truths. There should be 10 ground truth answers for each queston.
    It is a tensor of shape (batch_size, number_of_answers=10)
    """
    if len(net_answers) != len(gt_answers):
        raise ValueError("Length of net and gt answers not equal! net:{} gt:{}"
                         .format(len(net_answers), len(gt_answers)))
    
    # Get non-"<don't know>" predictions. <don't know> is index 0.
    # Offset +1 to correct for slicing
    net_predictions = net_answers[:,1:].argmax(dim=1)+1
    
    accs = []
    for net_pred, gt_ans in zip(net_predictions, gt_answers):
        net_pred = net_pred.item()
        
        acc_per_ans_subset = []
        for i in range(len(gt_ans)):
            ans_subset = torch.cat((gt_ans[:i], gt_ans[i+1:]))
            num_matching = (ans_subset == net_pred).sum().item()
            subset_acc = min(1, num_matching/3)
            acc_per_ans_subset.append(subset_acc)
        acc = sum(acc_per_ans_subset)/len(acc_per_ans_subset)
        accs.append(acc)
    return accs

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

def get_models(model_path):
    """Returns iterable of (filename, model state_dict) tuples"""
    # model_path is a .pth file. load and return it.
    if os.path.splitext(model_path)[1]==".pth":
        model_sd = torch.load(model_path, map_location='cpu')
        return [(model_path, model_sd)]
    
    # model_path points to folder, load modules lazily as generator.
    files = os.listdir(model_path)
    files = (os.path.splitext(f) for f in files)
    modelfiles = []
    for name, ext in files:
        if ext==".pth" and name.startswith("model"):
            modelfiles.append(name+ext)
    def module_gen():
        for m_filename in modelfiles:
            model_file = os.path.join(model_path, m_filename)
            model_sd = torch.load(model_file, map_location='cpu')
            yield (m_filename, model_sd)
    return module_gen()



def main(args):
    device = torch.device(args.device)
    
    # Get models
    model_sd_iter = get_models(args.model_path)
    
    # mapping wrappers
    with open(args.vocab_path,"rb") as f:
        vocab = pickle.load(f)
    with open(args.ans_path,"rb") as f:
        answers = pickle.load(f)
    
    # Dataloader
    data_loader = dataloader.get_loader(args.image_dir,
                                        args.annotation_path,
                                        args.question_path,
                                        vocab, answers,
                                        batch_size=args.batch_size,
                                        shuffle=False,
                                        num_workers=args.num_workers
                                       )
    
    # Model. Parameters are fixed.
    # Note: Expected values: len(vocab) = 8254, len(answers) = 3001
    net = model.VQAnet(len(vocab), embedding_size=128, lstm_size=512,
                       fc_size=1024, num_answers=len(answers))
    
    # Compute loss for validation too.
    # Don't reduce loss, so can manually weight by answers.
    criterion = nn.CrossEntropyLoss(reduce=False)
    
    # Compute validation for each model.
    stats = {}
    iterations = len(data_loader)
    for model_filename, model_sd in model_sd_iter:
        net.load_state_dict(model_sd)
        net.to(device)
        net.eval()
        
        
        start = time.time()
        total_loss = 0
        total_weight = 0
        total_acc = 0
        total_count = 0
        
        for i, (images, qn_idxs, ans_idxs) in enumerate(data_loader,1):
            images = images.to(device)
            if device.type=="cuda":
                qn_idxs = qn_idxs.cuda()
            elif device.type=="cpu":
                qn_idxs = qn_idxs.cpu()
            ans_unique, ans_weights, ans_repeats = unroll_ans(ans_idxs)
            ans_unique = ans_unique.to(device)
            ans_weights = ans_weights.type(torch.float).to(device)
            
            output = net(images, qn_idxs)
            
            # Compute loss
            if sum(ans_repeats) != 0:
                repeated_output = repeat(output, ans_repeats)
                losses = criterion(repeated_output, ans_unique)
                sum_losses = (losses*ans_weights).sum()
                sum_weights = ans_weights.sum()
                loss = sum_losses / sum_weights
                
                total_loss += sum_losses.item()
                total_weight += sum_weights.item()
            
            # Compute accuracy
            accs = evaluate_accuracy(output, ans_idxs)
            total_acc += sum(accs)
            total_count += len(accs)
            
            if i%args.log_period == 0:
                current_time = time.time() - start
                current_avg_loss = total_loss/total_weight
                current_avg_acc = total_acc/total_count
                print("Model: {:11}, Step: {:6}/{:6}, Loss:{:.5f}, Acc:{:.5f}, time:{:.3f}s".format(
                      model_filename, i, iterations, current_avg_loss, current_avg_acc, current_time),
                      flush=True, end="\r"
                     )
        
        current_time = time.time() - start
        current_avg_loss = total_loss/total_weight
        current_avg_acc = total_acc/total_count
        print("Model: {:11}, Step: {:6}/{:6}, Loss:{:.5f}, Acc:{:.5f}, time:{:.3f}s".format(
              model_filename, i, iterations, current_avg_loss, current_avg_acc, current_time),
              flush=True
             )
        stats[model_filename] = (current_avg_loss, current_avg_acc)
    print(stats)
    with open(args.stats_path, "wb") as f:
        pickle.dump(stats, f)
    return stats



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats_path', type=str, default='stats.pkl' , help="path to save dict of evaluation stats.")
    parser.add_argument('--model_path', type=str, default='models/' , help="path for trained model. Can be either a folder or a .pth file containing model's state_dict.")
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--ans_path', type=str, default='answers.pkl', help='path for answers wrapper')
    parser.add_argument('--image_dir', type=str, default='images/val2014', help='directory for images')
    parser.add_argument('--annotation_path', type=str,
                        default='annotations/v2_mscoco_val2014_annotations.json',
                        help='path for annotations json')
    parser.add_argument('--question_path', type=str,
                        default='questions/v2_OpenEnded_mscoco_val2014_questions.json',
                        help='path for questions json')
    parser.add_argument('--device', type=str, default="auto", choices=("auto","cpu","cuda"), help='device to use')
    parser.add_argument('--log_period', type=int, default=100,
                        help='Number of minibatches before progress is updated')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    # Note: seems like on windows, each worker uses about 2GB RAM.
    # But on ubuntu, they can apparently share memory and consume much less.
    # 8 seems ok on ubuntu
    
    args = parser.parse_args()
    
    if args.device is "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(args.device), flush=True)
    
    stats = main(args)