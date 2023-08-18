from tqdm import tqdm
import pickle
import json
import os
import math
import logging
import wandb

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MultilabelF1Score
from torch.optim.lr_scheduler import CosineAnnealingLR

import dgl
from dgl.nn.pytorch import GraphConv
from dgl.nn import GATConv
import networkx as nx
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def labels_to_tensor(labels, num_classes=3000):
    tensor = torch.zeros((1, num_classes))
    tensor[0, labels] = 1
    return tensor

### Feature extraction part:
def build_subgraphs(df,embedding, go_dict_len,save_path):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 所有唯一的蛋白
    unique_proteins = df['UniProtID1'].unique().tolist()
    
    no_neibour_protein = 0

    for protein in tqdm(unique_proteins):
        # 创建子图
        sub_g = dgl.DGLGraph()
        
        # 添加中心节点
        sub_g.add_nodes(1)
        center_node_id = 0
        node_mapping = {protein: center_node_id}
        
        # 获取与中心蛋白连接的邻居
        neighbors = df[(df['UniProtID1'] == protein)]
        
        # 为中心节点分配embedding
        center_embedding = embedding[protein]
        sub_g.nodes[center_node_id].data['feat'] = center_embedding.unsqueeze(0)

        # Assigning label to the centre protein
        protein_labels_center = eval(df[df['UniProtID1'] == protein]['GO_terms_associated_to_UniProtID1'].iloc[0])
        sub_g.nodes[center_node_id].data['label'] = labels_to_tensor(protein_labels_center,num_classes = go_dict_len)
        
        if len(neighbors) == 0:
            no_neibour_protein+=1
        else:
            # 添加邻居节点
            for idx, (_, row) in enumerate(neighbors.iterrows()):
                neighbor = row['UniProtID2']

                sub_g.add_nodes(1)
                sub_g.add_edges(center_node_id, idx + 1)
            
                node_mapping[neighbor] = idx + 1
                edge = (center_node_id, idx + 1)
                sub_g.edges[edge].data['weight'] = torch.tensor([row['Score']])
                
                # 为邻居节点分配embedding
                neighbor_embedding = embedding[neighbor]
                sub_g.nodes[idx + 1].data['feat'] = neighbor_embedding.unsqueeze(0)
                
                # Assigning label to the neighbor
                protein_labels_neigh = eval(neighbors[neighbors['UniProtID2'] == neighbor]['GO_terms_associated_to_UniProtID2'].iloc[0])
                sub_g.nodes[idx + 1].data['label'] = labels_to_tensor(protein_labels_neigh, num_classes = go_dict_len)
                
            # subgraphs.append(sub_g)
            dgl.save_graphs(f'{save_path}/{protein}.dgl',sub_g)
            

    print(f"In total has {len(unique_proteins)}  unique proteins but {no_neibour_protein} of them are ophan")

def plot_and_save_subgraphs(subgraphs_path, plot_number=None, path='subgraphs_plots'):
    # Create a directory to save subgraphs
    if not os.path.exists(path):
        os.makedirs(path)

    graph_list = [f for f in os.listdir(subgraphs_path)]
    
    if plot_number and len(graph_list) >= plot_number:
        import random
        random.seed(42)
        subgraphs = random.sample(graph_list, plot_number)
        for graph in tqdm(subgraphs):
            # Convert DGLGraph to NetworkX graph
            graph_path = os.path.join(subgraphs_path,graph)
            g,_ = dgl.load_graphs(graph_path)
            nx_g = g[0].to_networkx().to_undirected()
            
            # Draw using matplotlib
            plt.figure(figsize=(8, 8))
            pos = nx.spring_layout(nx_g)  # Layout for our graph
            nx.draw(nx_g, pos, with_labels=True, node_color=[[.7, .7, .7]])
            # Save the plot
            protein_name = graph.split('.')[0]
            plt.savefig(f"{path}/{protein_name}.png")
            plt.close()
            print(f"Saved: {path}/{protein_name}.png")
    else:
        print(f"There are only {len(graph_list)} graphs, but you asked for plotting {plot_number} graph(s). Please set a plot number smaller than {len(subgraphs)}")



###### model training part:
# Prepare datasets for models
class GraphDataset(Dataset):
    def __init__(self, protein_list,graph_path):
        super(GraphDataset, self).__init__()
        self.graphs = []
        for protein in protein_list:
            path = os.path.join(graph_path,protein) + '.dgl'
            graph, _ = dgl.load_graphs(path)
            self.graphs.append(graph[0])

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

# model architecture setup
class GCN(nn.Module):
    def __init__(self, num_input, num_hidden, num_output, dropout=0.5):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(num_input, num_hidden)
        self.conv2 = GraphConv(num_hidden, num_hidden)
        self.conv3 = GraphConv(num_hidden, num_output)
        
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, g, features):
        g = dgl.add_self_loop(g)
        
        # First convolution layer
        x = F.relu(self.conv1(g, features.float()))
        x = self.dropout(x)
        
        # Second convolution layer
        x = F.relu(self.conv2(g, x))
        x = self.dropout(x)
        
        # Third convolution layer
        x = self.conv3(g, x)
        
        return x
    

class GAT(nn.Module):
    def __init__(self, num_input, num_hidden, num_output, num_heads, dropout=0.5):
        super(GAT, self).__init__()
        
        self.conv1 = GATConv(in_feats=num_input, out_feats=num_hidden, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout)
        self.conv2 = GATConv(in_feats=num_hidden * num_heads, out_feats=num_hidden, num_heads=num_heads, feat_drop=dropout, attn_drop=dropout)
        self.conv3 = GATConv(in_feats=num_hidden * num_heads, out_feats=num_output, num_heads=1, feat_drop=dropout, attn_drop=dropout)
        
    def forward(self, g, features):
        g = dgl.add_self_loop(g)
        h = self.conv1(g, features.float())
        x = F.elu(h.view(h.size(0), -1))
        
        h = self.conv2(g, x)
        x = F.elu(h.view(h.size(0), -1))
        
        x = self.conv3(g, x).squeeze(1)
        
        return x

# train the model

def calculate_f1max(batched_graph, pred, label,f1_metric):
    node_counts = batched_graph.batch_num_nodes()
    cumulative_node_counts = torch.cumsum(node_counts, dim=0)
    center_indices = torch.cat([torch.tensor([0]).to(batched_graph.device), cumulative_node_counts[:-1]])

    center_labels = label[center_indices]
    center_logits = pred[center_indices]

    f1 = f1_metric(center_logits, center_labels.int())
    return f1

def test_f1max(batched_graph, pred, label,f1_metric):
    node_counts = batched_graph.batch_num_nodes().detach().cpu()
    cumulative_node_counts = np.insert(np.cumsum(node_counts), 0, 0)
    center_indices = cumulative_node_counts[:-1]
    return center_indices


def trainer(train_loader, val_loader, model, cfg, output_dim, device):
    
    early_stop=cfg.early_stop
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    l2 = cfg.l2
    lrs=cfg.lrs
    model_save_path = cfg.model_save_path
    f1_metric = MultilabelF1Score(num_labels=output_dim, average='micro').to(device)
    
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    # Define the optimization algorithm.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                         num_warmup_steps= 0,
    #                                         num_training_steps= len(train_loader)*n_epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    
    n_epochs, best_loss, step, early_stop_count = n_epochs, math.inf, 0, early_stop

    for epoch in range(n_epochs):
        model.train()  # Set the model to train mode.
        loss_record = []
        total_train_f1 = 0
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for batched_graph in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            # Move the data to device.
            batched_graph = batched_graph.to(device)
            b_labels = batched_graph.ndata['label'].squeeze(1).to(device) # b_labels batch_size * 1
            
            pred = model(batched_graph, batched_graph.ndata['feat'].to(device))
            loss = criterion(pred, b_labels)
            
            # predictions = (torch.sigmoid(pred) > 0.5)
            f1 = calculate_f1max(batched_graph, pred, b_labels,f1_metric)
            total_train_f1 += f1
            # Compute gradient(backpropagation).
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()                    # Update parameters.
            # scheduler.step()

            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'Train loss': loss.detach().item(),'Train f1': f'{f1:.3f}'})
            
            logging.info(f'Epoch [{epoch+1}/{n_epochs}] | step [{step}] | loss: {loss.detach().item()}')

        if lrs:
            scheduler.step()
        
        mean_train_loss = sum(loss_record)/len(loss_record)
        avg_train_f1 = total_train_f1/len(train_loader)

        wandb.log({'epoch': epoch,
                    'train_loss': mean_train_loss,
                    'train_f1': avg_train_f1,
                    'step': step
                    })

        ########### =========================== Evaluation=========================################
        logging.warning('###########=========================== Evaluating=========================################')

        model.eval()  # Set the model to evaluation mode.
        loss_record = []
        total_eval_f1 = 0
        preds = []
        labels = []

        val_pbar = tqdm(val_loader, position=0, leave=True)
        for batched_graph in val_pbar:

            with torch.no_grad():
                batched_graph = batched_graph.to(device)
                b_labels = batched_graph.ndata['label'].squeeze(1).to(device) # b_labels batch_size * 1
                
                pred = model(batched_graph, batched_graph.ndata['feat'].to(device))
                loss = criterion(pred, b_labels)

                f1 = calculate_f1max(batched_graph, pred, b_labels,f1_metric)
                total_eval_f1 += f1
                    
                # preds.append(pred.detach().cpu()[0].tolist())
                # labels.append(b_labels.detach().cpu()[0].tolist())

            loss_record.append(loss.item())

            val_pbar.set_description(f'Evaluating [{epoch + 1}/{n_epochs}]')
            val_pbar.set_postfix({'Valid loss': loss.detach().item(),'Valid f1': f'{f1:.3f}'})
            
            logging.info(f'Evaluating [{epoch + 1}/{n_epochs}] | step [{step}] | loss: {loss.detach().item()}')

        mean_valid_loss = sum(loss_record)/len(loss_record)# 一个batch的loss
        avg_val_f1 = total_eval_f1 / len(val_loader)

        wandb.log({'epoch': epoch, 
                    'valid_loss': mean_valid_loss,
                    'valid_f1': avg_val_f1,
                    'step': step
                    })

        logging.warning(f''' 
                        ***********************************
                        Epoch [{epoch + 1}/{n_epochs}]:
                        
                        Train loss: {mean_train_loss:.4f}, 
                        Valid loss: {mean_valid_loss:.4f}, 
                        Train f1: {avg_train_f1:.4f},
                        Valid f1: {avg_val_f1:.4f}
                        
                        ***********************************\n
                        ''')
        
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss

            torch.save({'model_state_dict': model.state_dict()}, 
                       f'{model_save_path}')  # Save the best model

            logging.warning('Saving model with loss {:.3f}...'.format(best_loss))

            early_stop_count = 0
        else:
            early_stop_count += 1
            logging.warning(f'{early_stop - early_stop_count} early stop points left for training')

        if early_stop_count >= early_stop:
            logging.warning('Model is not improving, so we halt the training session.')
            return


def predict(test_loader, model, device,usage):
    model.eval()  # Set the model to evaluation mode.
    preds = []
    labels = []
    for batched_graph in tqdm(test_loader):
        batched_graph = batched_graph.to(device)
        b_labels = batched_graph.ndata['label'].squeeze(1).to(device) # b_labels batch_size * 1
        pred = model(batched_graph, batched_graph.ndata['feat'].to(device))

        node_counts = batched_graph.batch_num_nodes()
        cumulative_node_counts = torch.cumsum(node_counts, dim=0)
        center_indices = torch.cat([torch.tensor([0]).to(batched_graph.device), cumulative_node_counts[:-1]])

        center_labels = b_labels[center_indices]
        center_logits = pred[center_indices]

        if usage == 'infer':
            preds.append(torch.sigmoid(center_logits).detach().cpu())
        else:
            preds.append(center_logits)
            labels.append(center_labels)
    if usage == 'infer':
        preds = torch.cat(preds, dim=0)
        labels = torch.zeros(preds.shape[0])
    else:
        preds = torch.cat(preds, dim=0)
        labels = torch.cat(labels, dim=0)

    return preds, labels


def predict_results(y_true, preds, name_list, save_path_predictions, save_path_metrics, save_num, label_num, go_dict, device, usage):
    if usage != 'infer':
        f1_metric = MultilabelF1Score(num_labels=label_num, average='micro').to(device)
        f1 = f1_metric(preds, y_true.int())
        print("F1 score: ", f1.detach().item())
        
        with open(save_path_metrics,'a+') as f:
            f.write(f"{save_num}: {f1.detach().item()}\n")
    
    else:
        pred_conf_list = []
        pred_goterms = []
        print('Recording the predictions...')
        for pred_value in tqdm(preds):
            pred_bin = torch.round(pred_value).int().numpy()
            
            pred_indices = np.where(pred_bin == 1)[0]
            pred_terms = [key for key, value in go_dict.items() if value in pred_indices]
            pred_goterms.append(pred_terms)
            
            pred_conf = pred_value[pred_indices]
            pred_conf_list.append(pred_conf.tolist())
        result = pd.DataFrame({
                'target_id': name_list,
                'pred_go': pred_goterms,
                'pred_pro': pred_conf_list
            })
        result.to_csv(save_path_predictions, sep='\t', index=False)

        print(f"Predictions saved to {save_path_predictions}")

        