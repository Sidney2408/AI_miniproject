import argparse
import os
import pickle
import tkinter as tkr
from tkinter import filedialog
from tkinter import messagebox
from tkinter import ttk

import nltk
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk

import mappings
import model


class Application(ttk.Frame):
    transform_display = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224)
    ])
    
    transform_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    def __init__(self, net, vocab, answers, device, master=None):
        super().__init__(master)
        self.pack(fill="both", expand=True)
        
        self.net = net
        self.vocab = vocab
        self.answers = answers
        self.device = device
        self.photoimage = None
        
        self.image_button = ttk.Button(self, text="Select image",
                                       command=self.select_image)
        self.image_button.pack(fill="x")
        self.image_frame = ttk.Frame(self, height=224, width=224)
        self.image_frame.pack_propagate(0)
        self.image_frame.pack(fill="both", expand=True)
        self.image_display = ttk.Label(self.image_frame)
        self.image_display.place(relx=.5, rely=.5, anchor='center')
        
        self.question_prompt = ttk.Label(self, text="Enter Question:")
        self.question_input = ttk.Entry(self)
        self.question_input.pack(fill="x")
        self.question_input.bind('<Return>', self.answer)
        
        self.answerstr = tkr.StringVar()
        self.answerstr.set("Answer will be displayed here")
        self.answer_box = tkr.Message(self, textvar=self.answerstr)
        self.answer_box.pack(fill="x", expand=True)
        self.answer_box.bind("<Configure>", lambda event: self.answer_box.configure(width=event.width-10))
        
        self.answer_button = ttk.Button(self, text="Ask Question", command=self.answer)
        self.answer_button.pack(fill="x")
        
        self.master.minsize(224,224)
    
    def select_image(self, event=None):
        image_path = filedialog.askopenfilename(filetypes = (("all files","*.*"),))
        with open(image_path, 'rb') as f:
            try:
                img = Image.open(f)
            except OSError as e:
                messagebox.showinfo("Error", "Cannot open image: {}".format(e))
                return
            img = img.convert('RGB')
        img = self.transform_display(img)
        
        self.photoimage = ImageTk.PhotoImage(img)
        self.image_display.configure(image=self.photoimage)
        
        self.tensorimage = self.transform_tensor(img).unsqueeze(0)
        self.tensorimage = self.tensorimage.to(self.device)
    
    def answer(self, event=None):
        qn = self.question_input.get()
        qn_idx = str2qn_idx(qn, self.vocab)
        qn_idx = qn_idx.to(self.device)
        packed_qn = torch.nn.utils.rnn.pack_sequence([qn_idx])
        
        output = self.net(self.tensorimage, packed_qn)
        output = output.squeeze(0)
        output = torch.nn.functional.softmax(output, dim=0)
        pred_prob, ans_idx = output.topk(5)
        
        answertext = []
        for prob, ans in zip(pred_prob, ans_idx):
            prob = prob.item()*100
            ans = ans.item()
            answertext.append("{:3.2f}%: {}".format(prob, self.answers.idx2ans[ans]))
        answertext = "\n".join(answertext)
        self.answerstr.set(answertext)



def str2qn_idx(qn, vocab):
    tokens = nltk.tokenize.word_tokenize(qn.lower())
    qn_idx = []
    qn_idx.append(vocab('<start>'))
    qn_idx.extend(vocab(token) for token in tokens)
    qn_idx.append(vocab('<end>'))
    qn_idx = torch.tensor(qn_idx, dtype=torch.long)
    return qn_idx

def main(args):
    device = torch.device(args.device)
    
    # mapping wrappers
    with open(args.vocab_path,"rb") as f:
        vocab = pickle.load(f)
    with open(args.ans_path,"rb") as f:
        answers = pickle.load(f)
    
    model_sd = torch.load(args.model_path, map_location='cpu')
    
    net = model.VQAnet(len(vocab), embedding_size=128, lstm_size=512,
                       fc_size=1024, num_answers=len(answers))
    net.load_state_dict(model_sd)
    net.to(device)
    net.eval()
    
    app = Application(net, vocab, answers, device, master=tkr.Tk())
    app.mainloop()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='model.pth' , help='path for trained model used for answering')
    parser.add_argument('--vocab_path', type=str, default='vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--ans_path', type=str, default='answers.pkl', help='path for answers wrapper')
    parser.add_argument('--device', type=str, default="auto", choices=("auto","cpu","cuda"), help='device to use')
    
    args = parser.parse_args()
    
    if args.device is "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: {}".format(args.device), flush=True)
    
    main(args)