import pandas as pd
import random
import torch
import time
import torchvision
import numpy as np
from gensim.models.word2vec import Word2Vec
from model_tbcnn import BatchProgramClassifier
from torch.autograd import Variable
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import sys
import copy

                
def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx: idx+bs]
    data, labels = [], []
    for _, item in tmp.iterrows():
        data.append(item[1])
        labels.append(item[2]-1)
    return data, torch.LongTensor(labels)
  
  
loss_function = torch.nn.CrossEntropyLoss()

def plot_training_statistics(train_stats, model_name):
    
    fig, axes = plt.subplots(2, figsize = (10,10))
    axes[0].plot(train_stats[f'{model_name}_Training_Loss'], label = f'{model_name}_Training_Loss')
    axes[0].plot(train_stats[f'{model_name}_Validation_Loss'], label = f'{model_name}_Validation_Loss')
    axes[1].plot(train_stats[f'{model_name}_Training_Acc'], label = f'{model_name}_Training_Acc')
    axes[1].plot(train_stats[f'{model_name}_Validation_Acc'], label = f'{model_name}_Validation_Acc')
    
    axes[0].set_xlabel("Number of Epochs"), axes[0].set_ylabel("Loss")
    axes[1].set_xlabel("Number of Epochs"), axes[1].set_ylabel("Accuracy in %")
    
    axes[0].legend(), axes[1].legend()

def fit_model(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS,ENCODE_DIM,LABELS,BATCH_SIZE,USE_GPU, embeddings):
    
    model = BatchProgramClassifier(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS,ENCODE_DIM,LABELS,BATCH_SIZE,
                                       USE_GPU, embeddings)
    if USE_GPU:
        model.cuda()
        
    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters, lr=0.0003)
    

    train_loss_ = []
    val_loss_ = []
    train_acc_ = []
    val_acc_ = []
    best_acc = 0.0

    print('Starting to train.....')
    # training procedure
    best_model = model
    for epoch in range(EPOCHS):
        start_time = time.time()

        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(train_data):
            train_batch = get_batch(train_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            train_inputs, train_labels = train_batch
            if USE_GPU:
               train_inputs, train_labels = train_inputs, train_labels.cuda()

            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train_inputs)

            loss = loss_function(output, Variable(train_labels))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.item()*len(train_inputs)

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc.item() / total)
            
        # validation epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(val_data):
            val_batch = get_batch(val_data, i, BATCH_SIZE)
            i += BATCH_SIZE
            val_inputs, val_labels = val_batch
            if USE_GPU:
               val_inputs, val_labels = val_inputs, val_labels.cuda()

            model.batch_size = len(val_labels)
            model.hidden = model.init_hidden()
            output = model(val_inputs)

            loss = loss_function(output, Variable(val_labels))

            # calc valing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == val_labels).sum()
            total += len(val_labels)
            total_loss += loss.item()*len(val_inputs)
            
        val_loss_.append(total_loss / total)
        val_acc_.append(total_acc.item() / total)
        end_time = time.time()
        if total_acc/total > best_acc:
           best_model = model
            
        print('[Epoch: %3d/%3d] Training Loss: %.4f, Validation Loss: %.4f,'
                  ' Training Acc: %.3f, Validation Acc: %.3f, Time Cost: %.3f s'
               % (epoch + 1, EPOCHS, train_loss_[epoch], val_loss_[epoch],
                 train_acc_[epoch], val_acc_[epoch], end_time - start_time))
        if val_loss_ > train_loss_:
           print('Validation Loss Decreased')
           print('Saving The Model')
                         
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')
                
    return  best_model , pd.DataFrame({f'{model_name}_Training_Loss' : train_loss_,
                        f'{model_name}_Training_Acc' : train_acc_,
                        f'{model_name}_Validation_Loss' : val_loss_,
                        f'{model_name}_Validation_Acc' : val_acc_})
                


if __name__ == '__main__':
    root = 'data/'
    train_data = pd.read_pickle('train/blocks.pkl')
    val_data = pd.read_pickle('dev/blocks.pkl')
    test_data = pd.read_pickle('test/blocks.pkl')

    word2vec = Word2Vec.load("train/embedding/node_w2v_128").wv
    embeddings = np.zeros((word2vec.syn0.shape[0] + 1, word2vec.syn0.shape[1]), dtype="float32")
    embeddings[:word2vec.syn0.shape[0]] = word2vec.syn0

    model_name = 'TBCNN' 
    HIDDEN_DIM = 300
    ENCODE_DIM = 128
    LABELS = 104
    EPOCHS = 10
    BATCH_SIZE = 32
    USE_GPU = True
    MAX_TOKENS = word2vec.syn0.shape[0]+1
    EMBEDDING_DIM = word2vec.syn0.shape[1]

    best_model , train_stats = fit_model(EMBEDDING_DIM,HIDDEN_DIM,MAX_TOKENS,ENCODE_DIM,LABELS,BATCH_SIZE,USE_GPU, embeddings)
    plot_training_statistics(train_stats, model_name)
                
    print('Starting to test.....')
    # testing procedure
    pred = []
    true = []
    test_acc = 0.0
    total_loss = 0.0
    total = 0.0
    i = 0
    model = best_model
    while i < len(test_data):
        test_batch = get_batch(test_data, i, BATCH_SIZE)
        i += BATCH_SIZE
        test_inputs, test_labels = test_batch
        if USE_GPU:
            test_inputs, test_labels = test_inputs, test_labels.cuda()
      
        model.batch_size = len(test_labels)
        model.hidden = model.init_hidden()
        output = model(test_inputs)

        loss = loss_function(output, Variable(test_labels))

        _, pred = torch.max(output.data, 1)
        true = test_inputs
        test_acc += (pred == test_labels).sum()
        total += len(test_labels)
        total_loss += loss.item() * len(test_inputs)
    print(['Testing results(Acc): %.3f' %(test_acc.item() / total)])    
