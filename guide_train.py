from scipy.sparse import data
import torch
import torch.nn as nn
import numpy as np
import scipy.sparse
import scipy.io
from utils import load_data_motif,normalize_adj,normalize
# from sklearn.metrics import  roc_auc_score
from metrics import auc_roc,auc_pr,precision,recall
from sklearn.preprocessing import MinMaxScaler

from datetime import datetime
import argparse
import random
from models import GUIDE

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--dataset', default='cora', choices=['acmv9','citationv1','cora','dblpv7','pubmed'], help='dataset')
parser.add_argument('--hidden_dim', type=int, default=32, help='dimension of hidden embedding (default: 64)')
parser.add_argument('--epoch', type=int, default=200, help='Training epoch')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
parser.add_argument('--alpha', type=float, default=0.3, help='balance parameter')
parser.add_argument('--beta', type=float, default=0.3, help='loss parameter')
# parser.add_argument('--device', default='cuda', type=str, help='cuda/cpu')
parser.add_argument('--seed', type=int, default=2021, help='Training epoch')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print('dataset:', args.dataset)
device = torch.device("cuda:6" if args.cuda else "cpu")
print(device)

dataname = args.dataset+'_both_motif'
adj, adj_norm, X, labels, motifs, _ = load_data_motif(dataname)

scaler1 = MinMaxScaler()  
scaler2 = MinMaxScaler()  
X_norm = scaler1.fit_transform(X)  
motifs_norm = scaler2.fit_transform(motifs)  
# X_norm = X
# motifs_norm = motifs

adj_norm = torch.FloatTensor(adj_norm).to(device)
adj_label = torch.FloatTensor(adj).to(device)
attrs = torch.FloatTensor(X_norm).to(device)
motifs_norm = torch.FloatTensor(motifs_norm).to(device)
# labels

emb_size=args.hidden_dim
hidden1=emb_size*2
hidden2=hidden1*2

model = GUIDE(attrs.shape[1],motifs_norm.shape[1],hidden2,hidden1,emb_size,args.dropout,alpha=args.alpha).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr,weight_decay=args.weight_decay)


results = []

def loss_func_GUIDE(X, X_rec, motifs_feat, S_hat, beta):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X - X_rec, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # high-order structure reconstruction loss
    diff_structure = torch.pow(motifs_feat - S_hat, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost =  beta * attribute_reconstruction_errors + (1-beta) * structure_reconstruction_errors
    return cost, structure_cost, attribute_cost

for epoch in range(1,args.epoch+1):
    model.train()
    optimizer.zero_grad()
    X_hat, S_hat = model(attrs, motifs_norm, adj_norm) 
    loss, struct_loss, feat_loss = loss_func_GUIDE(attrs, X_hat, motifs_norm, S_hat, args.beta)
    l = torch.mean(loss)
    l.backward()
    optimizer.step()
    # print("Epoch:", '%04d' % (epoch), 
    #         "train_loss=", "{:.5f}".format(l.item()), 
    #         "train/struct_loss=", "{:.5f}".format(struct_loss.item()),
    #         "train/feat_loss=", "{:.5f}".format(feat_loss.item()))
    
    # if epoch%5 == 0 or epoch == args.epoch - 1:
    with torch.no_grad():
        model.eval()
        X_hat, S_hat = model(attrs, motifs_norm, adj_norm) 
        loss, struct_loss, feat_loss = loss_func_GUIDE(attrs, X_hat, motifs_norm, S_hat, args.beta)
        score = loss.detach().cpu().numpy()
        presicions = precision(score, labels,(50,100,150))
        recalls = recall(score, labels,(50,100,150))
        auc = auc_roc(score, labels)
        pr = auc_pr(score, labels)
        print("Epoch: {:4d}, roc {:.4f}, pr {:.4f}, Precision@K: [50,100,150] {:.4f}{:.4f}{:.4f}, Recall@K: [50,100,150] {:.4f}{:.4f}{:.4f}".format(epoch,auc,pr,presicions[0],presicions[1],presicions[2],recalls[0],recalls[1],recalls[2]))


# test
print('Evaluation:')

with torch.no_grad():
    model.eval()
    X_hat, S_hat = model(attrs, motifs_norm, adj_norm) 
    loss, struct_loss, feat_loss = loss_func_GUIDE(attrs, X_hat, motifs_norm, S_hat, args.beta)
    score = loss.detach().cpu().numpy()
    auc = auc_roc(score,labels)
    pr = auc_pr(score,labels)
    print('AUC:',auc)
    print('PR:',pr)
    print("Precision@K:", '[50,100,150]:', precision(score, labels,(50,100,150)), 
          "Recall@K:", '[50,100,150]:', recall(score, labels,(50,100,150)))
