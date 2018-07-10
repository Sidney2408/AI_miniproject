import torch
import torch.nn as nn
import torchvision.models

class Passthrough(nn.Module):
    def forward(self, inp):
        return inp

class VQAnet(nn.Module):
    
    def __init__(self, num_vocab, embedding_size, lstm_size, fc_size, num_answers):
        super().__init__()
        # Image network
        self.resnet = torchvision.models.resnet152(pretrained=True)
        resnet_size = self.resnet.fc.in_features
        self.resnet.fc = Passthrough() # decapitate resnet.
        # Prevent training. Use as feature extractor.
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.eval()
        
        # Question network
        self.embedding = nn.Embedding(num_embeddings=num_vocab,
                                      embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=lstm_size
                           )
        self.lstm_size = lstm_size
        
        # After concating both image activations and question activations,
        # dropout, then put through fully connected layer to get features.
        # (Note: Dropouts are applied on activations.)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(resnet_size+lstm_size, fc_size)
        self.relu = torch.nn.ReLU()
        
        # Final fully connected layer as classifier.
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(fc_size, num_answers)
    
    def forward(self, images, questions):
        # resnet outputs (batchsize*2048)
        image_activations = self.resnet(images)
        
        # questions must be packed sequences. Tensors not allowed.
        embed_qns = self.embedding(questions.data)
        embed_qns = nn.utils.rnn.PackedSequence(embed_qns, questions.batch_sizes)
        # Compute lstm final hidden state. hiddenstate is (batchsize*lstm_size)
        # Get device dynamically for initial states.
        device = torch.device("cuda" if next(self.parameters()).is_cuda else "cpu")
        # Initial hidden state
        h0 = torch.zeros((1, embed_qns.batch_sizes[0], self.lstm_size),
                         device=device)
        # Initial memory cell
        c0 = torch.zeros((1, embed_qns.batch_sizes[0], self.lstm_size),
                         device=device)
        output, (h_final, c_final) = self.lstm(embed_qns, (h0,c0))
        qn_activations = h_final[-1,:,:]
        
        # 1st fc layer.
        activations = torch.cat((image_activations, qn_activations), dim=1)
        output = self.dropout1(activations)
        output = self.fc1(output)
        output = self.relu(output)
        # 2nd fc layer (classifier)
        output = self.dropout2(output)
        output = self.fc2(output)
        
        return output
    
    def trainable_parameters(self):
        """modified version of parameters() to return non-resnet parameters"""
        for name, param in self.named_parameters():
            if name.startswith("resnet"):
                continue
            yield param
    
    def train(self, mode=True):
        super().train(mode)
        # Force resnet to be in eval mode.
        self.resnet.eval()
    
    def state_dict(self):
        # Remove resnet state from state_dict, since it should never change.
        state_dict = super().state_dict()
        resnet_states = [s for s in state_dict.keys() if s.startswith("resnet")]
        
        for state in resnet_states:
            del state_dict[state]
        
        return state_dict
    
    def load_state_dict(self, new_state_dict):
        # new_state_dict should only contain non-resnet state,
        # so update the current state_dict (which has resnet state) with it.
        # Because state_dict must contain all keys of the model.
        state_dict = super().state_dict()
        state_dict.update(new_state_dict)
        super().load_state_dict(state_dict)