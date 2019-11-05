import torch
import torch.nn as nn
import torch.nn.functional as F

class Text2BBoxesModel(nn.Module) :
    def __init__(self, hidden_size, labels, batch_size,device) :
        super(Text2BBoxesModel, self).__init__()
        self.device = device
        self.num_categories = labels[max(labels, key=labels.get)] + 1
        self.lstm = nn.LSTMCell(self.num_categories, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.output_label = nn.Linear(hidden_size, self.num_categories)
     #   self.output_coords = nn.Linear(hidden_size + self.num_categories, )
        self.batch_size = batch_size
        self.SOS = self.num_categories - 2
        self.EOS = self.num_categories - 1
        
       # self.output_coords = nn.Linear(hidden_size)
    
        
    def init_hidden_states(self, input_embedding) :
        self.hidden = input_embedding.to(self.device)
        self.cell = torch.randn_like(input_embedding, requires_grad=True).to(self.device)

    def forward(self, caption_embedding, max_length) :
#         output_seq = torch.empty((self.sequence_len, 
#                                   self.batch_size, 
#                                   self.num_categories))
#         output_list = [torch.tensor([0.0])] * batch_size
        self.init_hidden_states(caption_embedding)
        t = 0
        start_tensor = self.index_to_one_hot(self.SOS)
        pred_labels = start_tensor.repeat(self.batch_size,1).float()
        pred_labels.requires_grad_(True)
        pred_labels = pred_labels.to(self.device)
        outputs = []
        while(t < max_length) :
            inputs = pred_labels
            self.hidden, self.cell = self.lstm(inputs, (self.hidden, self.cell))
            pred_labels = self.softmax(self.output_label(self.hidden))
            t+=1
            outputs.append(pred_labels)
        return torch.stack(outputs, dim=0)
    
    def loss_function_labels(self, preds, labels) :
        return F.cross_entropy(preds, labels)
    
    def one_hot_to_label(self, label) :
        value, index = torch.max(label[0], 0)
        return index

    def index_to_one_hot(self, index) :
        labels_one_hot = torch.Tensor(self.num_categories).zero_()
        labels_one_hot[index] = 1
        return labels_one_hot

