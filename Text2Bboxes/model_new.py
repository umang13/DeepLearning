import torch
import torch.nn as nn
import torch.nn.functional as F
from MDN import MDN

class Text2BBoxesModel(nn.Module) :
    def __init__(self, hidden_size, labels, batch_size,device) :
        super(Text2BBoxesModel, self).__init__()
        self.device = device
        self.num_categories = labels[max(labels, key=labels.get)] + 1
        # input size is num_classes + bbox_coordinates
        self.lstm = nn.LSTMCell(self.num_categories + 4, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.output_label = nn.Linear(hidden_size, self.num_categories)
        self.output_coords = nn.Linear(hidden_size + self.num_categories, 2)
        self.output_size = nn.Linear(hidden_size + self.num_categories + 2, 2)
        self.batch_size = batch_size
        self.SOS = self.num_categories - 2
        self.EOS = self.num_categories - 1
        self.num_mixtures = 5
        self.mdn_xy_model = MDN(2, self.num_mixtures)
        self.mdn_wh_model = MDN(2, self.num_mixtures)
        
       # self.output_coords = nn.Linear(hidden_size)
    
        
    def init_hidden_states(self, input_embedding) :
        self.hidden = input_embedding.to(self.device)
        self.cell = torch.randn_like(input_embedding, requires_grad=True).to(self.device)

    def forward(self, caption_embedding, max_length) :

        self.init_hidden_states(caption_embedding)
        t = 0
        start_tensor = self.index_to_one_hot(self.SOS)
        pred_labels = start_tensor.repeat(self.batch_size,1).float()
        pred_labels.requires_grad_(True)
        bbox_tensor = torch.Tensor([0,0,0,0])
        bbox_tensor = bbox_tensor.repeat(self.batch_size, 1)
        bbox_tensor.requires_grad_(True)
        pred_labels = pred_labels.to(self.device)
        bboxes = bbox_tensor.to(self.device)
        outputs = []
        while(t < max_length) :
            inputs = torch.cat([pred_labels, bboxes], 1)
            self.hidden, self.cell = self.lstm(inputs, (self.hidden, self.cell))
            pred_labels = self.softmax(self.output_label(self.hidden))
            pred_bboxes_coords = self.output_coords(torch.cat([self.hidden, pred_labels], 1))
            theta_xy = self.mdn_xy_model(pred_bboxes_coords)
            pred_bboxes_sizes = self.output_size(torch.cat([self.hidden, pred_labels, pred_bboxes_coords], 1))
            theta_wh = self.mdn_wh_model(pred_bboxes_sizes)
            print("====>", theta_xy[0].shape, theta_xy[1].shape, theta_xy[2].shape, theta_wh[0].shape, theta_wh[1].shape, theta_wh[2].shape)
            t+=1
            bboxes = torch.cat([pred_bboxes_coords, pred_bboxes_sizes], 1)
            output = torch.cat([pred_labels, bboxes], 1)
            outputs.append(output)
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

