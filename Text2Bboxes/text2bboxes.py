import torch
from pycocotools.coco import COCO
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import sys
from model_new import Text2BBoxesModel
from dataset import get_data_loader_and_cats


dataType='train2017'
dataDir = './Datasets/coco/images/{}/'.format(dataType)
annFile_Detection ='./Datasets/coco/annotations/instances_{}.json'.format(dataType)
annFile_Caption ='./Datasets/coco/annotations/captions_{}.json'.format(dataType)
batch_size = 256
coco = COCO(annFile_Detection)
coco_captions = COCO(annFile_Caption)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embed_size = 768
epochs = 30
learning_rate = 1e-3
beta1 = 0.9
beta2 = 0.999
ckpt_file_path = "./outputs/checkpoint_latest.ckpt"
print(device)


dataloader, total_categories = get_data_loader_and_cats(annFile_Caption, annFile_Detection, batch_size, True)
start = time.time()
num_classes = total_categories[max(total_categories, key=total_categories.get)] + 1
model = Text2BBoxesModel(embed_size, total_categories, batch_size, device)
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, betas=(beta1, beta2))
if(os.path.exists(ckpt_file_path)) :
    checkpoint = torch.load(ckpt_file_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model loaded with training up to {} epochs and {} minibatches".format(checkpoint['epoch'], checkpoint['count']))

model = model.to(device)
for epoch in range(epochs) : 
    count = 0
    for _, captions_mini_batch, bboxes, categories, lengths in dataloader :
        captions_mini_batch = captions_mini_batch.to(device)
        bboxes = bboxes.to(device)
        categories = categories.to(device)
        loss = 0
        optimizer.zero_grad()
        max_length = max(lengths)
        lengths = torch.Tensor(lengths)
        outputs = model(captions_mini_batch, max_length)
        print(outputs.shape)

        categories = categories.permute(1, 0)
        seqlen = categories.shape[0] 
        for i in range(0,seqlen) :
            loss += F.cross_entropy(pred_labels[i, :, :], categories[i, :].long())
        loss /= seqlen
        loss.backward()
        optimizer.step()
        count += 1            
        if(count % 100 == 0) :
            checkpoint = {'model_state_dict': model.state_dict(),
                          'optimizer_state_dict' : optimizer.state_dict(),
                          'epoch'  : epoch,
                          'count'  : count}
            torch.save(checkpoint, ckpt_file_path)
            print("After processing {} minibatches in epoch {} , loss is {}".format(count, epoch, loss))
        if(epoch == 0) :
            break


    print("Loss after epoch {} : {}".format(epoch, loss))
    print("Predictions on validation sets")
    with torch.no_grad() :
        dataType = 'val2017'
        dataDir = './Datasets/coco/images/{}/'.format(dataType)
        annFile_Detection ='./Datasets/coco/annotations/instances_{}.json'.format(dataType)
        annFile_Caption ='./Datasets/coco/annotations/captions_{}.json'.format(dataType)
        val_dataloader, _ = get_data_loader_and_cats(annFile_Caption, annFile_Detection, batch_size, True)
        val_loss = 0
        for caption_txt, caption, bboxes, categories, lengths in val_dataloader :
            caption = caption.to(device)
            bboxes = bboxes.to(device)
            categories = categories.to(device)
            max_length = max(lengths)
            pred_labels = model(caption, max_length)
            vals, idx = torch.max(pred_labels, 2)
            categories = categories.permute(1, 0)
            seqlen = categories.shape[0] 
            for i in range(0,seqlen) :
                val_loss += F.cross_entropy(pred_labels[i, :, :], categories[i, :].long())
            val_loss /= seqlen
        print("Loss on validaton set : {}".format(val_loss))
        print("Predicted labels shape : {}, Ground truth labels shape {}".format(idx.shape, categories.shape))
        # Print a sample of 10 captions from the val set
        for i in range(10) :
            print("Text input : {}".format(caption_txt[i]))
            print("Predicted labels {}, Ground truth labels {}".format(idx[:,i], categories[:,i]))


