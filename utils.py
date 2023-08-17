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
import esm

from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=UserWarning)

### Dataset preparation part:
class ESMDataset(Dataset):
    def __init__(self,row):
        super().__init__()
        # self.seq = row[f'{datatype}_seq']
        # self.aa = row[f'{datatype}_aa_list']
        self.mt_seq = row['mt_seq']
        self.mt_aa = row['mt_aa_list']
        self.wt_aa = row['wt_aa_list']
        self.gene_id = row['record_id']
        self.aa_index = row['aa_index']
        self.label = row['label']
    def __len__(self):
        return len(self.label)
    def __getitem__(self, idx):
        return (self.label[idx],self.mt_seq[idx],self.mt_aa[idx],self.wt_aa[idx], self.gene_id[idx],eval(self.aa_index[idx]))
        
def collate_fn(batch):
    labels, sequences, mt_aa, wt_aa, gene_id, aa_index = zip(*batch)
    return list(zip(labels, sequences)), wt_aa, mt_aa, gene_id, aa_index, sequences

def get_logits(total_logits,aa,esm_dict):
    softmax = nn.Softmax(dim=-1)
    aa_id = [esm_dict[x] - 4 if x != '-' else -1 for x in eval(aa)] # -1 since the "-" is the last one in the logits matrix
    
    batch_aa_id = torch.arange(len(aa_id))
    logits = softmax(total_logits)[batch_aa_id, aa_id]
    return logits
    
def generate_embeds_and_save(df, wt_seq, save_path, model_selection, device, batch_size=2):
    if model_selection =='esm1v':
        esm_model, esm_alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    elif model_selection == 'esm1b':
        esm_model, esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    elif model_selection == 'esm2':
        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif model_selection == "esm2_3b":
        esm_model, esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif model_selection == "esm2_15b":
        esm_model, esm_alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    
    batch_converter = esm_alphabet.get_batch_converter()
    esm_dict = esm_alphabet.tok_to_idx
    esm_model = esm_model.to(device) # move your model to GPU
    
    wt_batch_tokens = batch_converter([("protein",wt_seq)])
    
    mt_dataset = ESMDataset(df)
    mt_dataloader = DataLoader(mt_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn, drop_last=False)

    final_result = {}
    
    for batch in tqdm(mt_dataloader,total=len(mt_dataloader)):
        batch_labels, _, mt_batch_tokens = batch_converter(batch[0])
        wt_aa, mt_aa, mkk_name, aa_index_list = batch[1], batch[2], batch[3], batch[4]

        with torch.no_grad():
            
            wt_results = esm_model(wt_batch_tokens[2].to(device), repr_layers=[33]) # 1, seq_len, 1280
            mt_results = esm_model(mt_batch_tokens.to(device), repr_layers=[33]) # batch_size, seq_len, 1280
            
            for batch_idx, (aa_index, wt_list, mt_list, mkk, label) in enumerate(zip(aa_index_list, wt_aa, mt_aa, mkk_name, batch_labels)):

                aa_index = torch.tensor(aa_index).to(device)
                batch_indices = torch.arange(len(aa_index))
                # representation's first and last token are special tokens
                # Here the indices starts from 1, rather than 0, so we do not need to -1
                wt_repr = wt_results["representations"][33][0, aa_index,:].mean(dim=0).detach().cpu() 
                mt_repr = mt_results["representations"][33][batch_idx, aa_index,:].mean(dim=0).detach().cpu()

                total_logits = wt_results['logits'][0,:,list(range(4,24)) + [-3]] # batch_size, 1+seq_len+1, 20 common aa + 1 "-"
                total_logits = total_logits[aa_index,:] # mutation_len, 21

                wt_logits = get_logits(total_logits, wt_list, esm_dict) # mutation_len
                mt_logits = get_logits(total_logits, mt_list, esm_dict)
                logits = torch.sum(torch.log(mt_logits/wt_logits)).detach().cpu()
                
                final_result[mkk] = {"wt_emb": wt_repr, "mt_emb":mt_repr, "logits":logits, 'label': label}

    # save your embeddings
    torch.save(final_result,f'{save_path}') # save the embeddings
    del esm_model

def generate_whole_embeds_and_save(df, save_path, model_selection, device, batch_size=2):
    if model_selection =='esm1v':
        esm_model, esm_alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    elif model_selection == 'esm1b':
        esm_model, esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    elif model_selection == 'esm2':
        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif model_selection == "esm2_3b":
        esm_model, esm_alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    elif model_selection == "esm2_15b":
        esm_model, esm_alphabet = esm.pretrained.esm2_t48_15B_UR50D()
    
    batch_converter = esm_alphabet.get_batch_converter()
    esm_dict = esm_alphabet.tok_to_idx
    esm_model = esm_model.to(device) # move your model to GPU
    
    wt_seq = "MPKKKPT-P-IQLNP"
    wt_batch_tokens = batch_converter([("protein",wt_seq)])
    
    mt_dataset = ESMDataset(df)
    mt_dataloader = DataLoader(mt_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn, drop_last=False)

    final_result = {}
    
    for batch in tqdm(mt_dataloader,total=len(mt_dataloader)):
        batch_labels, _, mt_batch_tokens = batch_converter(batch[0])
        wt_aa, mt_aa, mkk_name, aa_index_list = batch[1], batch[2], batch[3], batch[4]
        mt_seq = batch[5]
        with torch.no_grad():
            
            wt_results = esm_model(wt_batch_tokens[2].to(device), repr_layers=[33]) # 1, seq_len, 1280
            mt_results = esm_model(mt_batch_tokens.to(device), repr_layers=[33]) # batch_size, seq_len, 1280
            
            for batch_idx, (aa_index, mtseq_list, mkk, label) in enumerate(zip(aa_index_list, mt_seq, mkk_name, batch_labels)):

                aa_index = torch.tensor(aa_index)
                # batch_indices = torch.arange(len(aa_index))
                
                # representation's first and last token are special tokens
                # Here the indices starts from 1, rather than 0, so we do not need to -1
                wt_repr = wt_results["representations"][33][0,1:-1,:].detach().cpu()
                mt_repr = mt_results["representations"][33][batch_idx,1:-1,:].detach().cpu()
                total_logits = wt_results['logits'][0,:,list(range(4,24)) + [-3]] # batch_size, 1+seq_len+1, 20 common aa + 1 "-"
                
                wt_list = str(list(wt_seq))
                mt_list = str(list(mtseq_list))
                wt_logits = get_logits(total_logits, wt_list, esm_dict) # mutation_len
                mt_logits = get_logits(total_logits, mt_list, esm_dict)
                logits = torch.log(mt_logits/wt_logits).unsqueeze(1).detach().cpu()
                
                final_result[mkk] = {"mt_emb":mt_repr, 'logits': logits, 'aa_index': aa_index,'label': label}
                
    final_result['wt_emb'] = wt_repr

    # save your embeddings
    torch.save(final_result,f'{save_path}') # save the embeddings
    del esm_model


# # fetch the embeddings
def unpickler(ds_name, input_type="mutant",logits=False):
    # original unpickler
    path = f'{ds_name}'

    pt_embeds = torch.load(path)
    
    name_list = []
    embedding_list = []
    logits_list = []
    label_list = []
    for name,content in pt_embeds.items():
        name_list.append(name)
        logits_list.append(content['logits'])
        label_list.append(content['label'])
        if input_type == "mutant":
            embedding_list.append(content['mt_emb'])
        elif input_type == "twin":
            embedding= torch.cat((content['wt_emb'], content['mt_emb']), dim=-1)
            embedding_list.append(embedding)

    data_X = torch.stack(embedding_list) 
    
    if logits is False:
        return data_X, label_list, name_list
    elif logits == "include":
        logits_tensor = torch.tensor(logits_list).unsqueeze(1)
        data_X = torch.hstack((data_X, logits_tensor))
        return data_X, label_list, name_list
    elif logits == "only":
        return logits_list, label_list, name_list

###### model training part:
# Prepare datasets for models
class MLPDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()

        self.seq = torch.tensor(X)
        self.label = torch.tensor(y)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, index):
        return self.seq[index], self.label[index]

# model architecture setup
class MLPClassifier_LeakyReLu(nn.Module):
    """Simple MLP Model for Classification Tasks.
    """

    def __init__(self, num_input, num_hidden, num_output,dropout):
        super(MLPClassifier_LeakyReLu, self).__init__()

        # Instantiate an one-layer feed-forward classifier
        self.hidden = nn.Linear(num_input, num_hidden) # 300,1280 -> 5,1280  10,1280 20,1280
        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            # nn.LeakyReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden, int(num_hidden/2)),
            nn.Linear(int(num_hidden/2), num_output)
        )
        self.softmax = nn.Softmax(dim=1) 
        
    def forward(self, x):
        x = self.hidden(x)
        x = self.predict(x)
            
        return x

# train the model
def flat_accuracy(pred_flat, labels):
    equal_count = torch.sum(pred_flat == labels)
    acc_rate = equal_count.item() / pred_flat.size(0)
    mcc = matthews_corrcoef(labels.tolist(), pred_flat.tolist())
    return acc_rate, mcc


def trainer(train_loader, val_loader, model, weight, cfg, device):
    
    early_stop=cfg.early_stop
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    model_save_path = cfg.model_save_path
    label_num = int(cfg.label_num)
    weights= weight.to(device)
    
    if cfg.loss_fn == 'mse':
        criterion = nn.MSELoss(reduction='mean')
    elif cfg.loss_fn == 'bce':
        criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=weights) 
    elif cfg.loss_fn == 'ce':
        criterion = nn.CrossEntropyLoss(reduction='sum',  weight=weights)
    else:
        raise ValueError('loss function not supported, please choose from mse, bce or ce')
        
    # Define the optimization algorithm.
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)
    # scheduler = get_linear_schedule_with_warmup(optimizer,
    #                                         num_warmup_steps= 0,
    #                                         num_training_steps= len(train_loader)*n_epochs)

    n_epochs, best_loss, step, early_stop_count = n_epochs, math.inf, 0, early_stop

    for epoch in range(n_epochs):
        model.train()  # Set the model to train mode.
        loss_record = []
        total_train_accuracy = 0
        total_train_mcc = 0
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for batch in train_pbar:
            optimizer.zero_grad()               # Set gradient to zero.
            # Move the data to device.
            b_seq, b_labels = (t.to(device) for t in batch)

            pred = model(b_seq.float()) # b_seq  batch_size * 2560, pred batch_size * label_num
            b_labels = b_labels.float() # b_labels batch_size * 1
            if cfg.loss_fn == 'mse':
                loss = criterion(pred[:,0], b_labels) # MSELoss
                acc = (1/loss.detach().item())
                mcc = 0
            elif cfg.loss_fn == 'bce':
                # b_labels_one_hot = torch.stack((b_labels, 1-b_labels), dim=1)
                # loss = criterion(pred, b_labels_one_hot)
                # pred_flat = torch.argmax(pred,dim=1)
                
                loss = criterion(pred.squeeze(), b_labels) # BCEWithLogitsLoss output dim=1
                # loss = criterion(pred[:,0], b_labels) # BCEWithLogitsLoss or BCELoss, output dim should be 2
                pred_flat = torch.round(torch.sigmoid(pred.squeeze())).int()
                
                acc, mcc = flat_accuracy(pred_flat, b_labels.long())
            elif cfg.loss_fn == 'ce':
                loss = criterion(pred, b_labels.long()) # CrossEntropy, output dim should be 2
                pred_flat = torch.argmax(pred,dim=1)
                acc, mcc = flat_accuracy(pred_flat, b_labels.long())
                
            total_train_accuracy += acc
            total_train_mcc += mcc

            # Compute gradient(backpropagation).
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()                    # Update parameters.
            # scheduler.step()

            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            train_pbar.set_postfix({'Train loss': loss.detach().item(),'Train acc': f'{acc:.3f}'})
            
            logging.info(f'Epoch [{epoch+1}/{n_epochs}] | step [{step}] | loss: {loss.detach().item()}')
            
        mean_train_loss = sum(loss_record)/len(loss_record)
        avg_train_accuracy = total_train_accuracy/len(train_loader)
        avg_train_mcc = total_train_mcc/len(train_loader)
        
        wandb.log({'epoch': epoch,
                       'train_loss': mean_train_loss,
                       'train_acc': avg_train_accuracy,
                       'train_mcc': avg_train_mcc,
                       'step': step
                       })

        ########### =========================== Evaluation=========================################
        logging.warning('###########=========================== Evaluating=========================################')

        model.eval()  # Set the model to evaluation mode.
        loss_record = []
        total_eval_accuracy = 0
        total_eval_mcc = 0
        preds = []
        labels = []

        val_pbar = tqdm(val_loader, position=0, leave=True)
        for batch in val_pbar:

            # Move your data to device.
            b_seq, b_labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                pred = model(b_seq.float())
                b_labels = b_labels.float()
                
                if cfg.loss_fn == 'mse':
                    loss = criterion(pred[:,0], b_labels) # MSELoss
                    acc = (1/loss.detach().item())
                    mcc=0
                elif cfg.loss_fn == 'bce':
                    # b_labels_one_hot = torch.stack((b_labels, 1-b_labels), dim=1)
                    # loss = criterion(pred, b_labels_one_hot) # BCEWithLogitsLoss or BCELoss, output dim should be 2
                    # pred_flat = torch.argmax(pred,dim=1)
                    
                    # loss = criterion(pred[:,0], b_labels) #dim=2

                    loss = criterion(pred.squeeze(), b_labels) # BCEWithLogitsLoss output dim=1
                    pred_flat = torch.round(torch.sigmoid(pred.squeeze())).int()
                    
                    acc, mcc = flat_accuracy(pred_flat, b_labels.long())
                elif cfg.loss_fn == 'ce':
                    loss = criterion(pred, b_labels.long()) # CrossEntropy, output dim should be 2
                    pred_flat = torch.argmax(pred,dim=1)
                    acc, mcc = flat_accuracy(pred_flat, b_labels.long())
                
                total_eval_accuracy += acc
                total_eval_mcc += mcc

                preds.extend(pred[:, 0].tolist())
                labels.extend(b_labels.int().tolist())

            loss_record.append(loss.item())

            val_pbar.set_description(f'Evaluating [{epoch + 1}/{n_epochs}]')
            val_pbar.set_postfix({'Valid loss': loss.detach().item(),'Valid acc': f'{acc:.3f}'})
            
            logging.info(f'Evaluating [{epoch + 1}/{n_epochs}] | step [{step}] | loss: {loss.detach().item()}')
            
            
        if epoch == cfg.record_epoch:
            # For selecting the best MCC threshold
            threshold = pd.DataFrame({
                    'label': labels,
                    'prediction': preds
                    })
            threshold.to_csv(f"./threshold_pick_{cfg.record_epoch}.txt", sep='\t', index=False)

        mean_valid_loss = sum(loss_record)/len(loss_record)# 一个batch的loss
        avg_val_accuracy = total_eval_accuracy / len(val_loader) # 一个batch的acc
        avg_val_mcc = total_eval_mcc / len(val_loader)
        
        wandb.log({'epoch': epoch, 
                       'valid_loss': mean_valid_loss,
                       'valid_acc': avg_val_accuracy,
                       'valid_mcc': avg_val_mcc,
                       'step': step
                       })
        
        logging.warning(f''' 
                        ***********************************
                        Epoch [{epoch + 1}/{n_epochs}]:
                        
                        Train loss: {mean_train_loss:.4f}, 
                        Valid loss: {mean_valid_loss:.4f}, 
                        Train acc: {avg_train_accuracy:.4f}, 
                        Valid acc: {avg_val_accuracy:.4f},
                        Train mcc: {avg_train_mcc:.4f},
                        Valid mcc: {avg_val_mcc:.4f}
                        
                        ***********************************\n
                        ''')
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss

            logging.warning('Saving model with loss {:.3f}...'.format(best_loss))
            torch.save({'model_state_dict': model.state_dict()}, 
                       f'{model_save_path}')  # Save the best model

            early_stop_count = 0
        else:
            early_stop_count += 1
            logging.warning(f'{early_stop - early_stop_count} early stop points left for training')

        if early_stop_count >= early_stop:
            logging.warning('Model is not improving, so we halt the training session.')
            return


def predict(test_loader, model, label_num, device):
    model.eval()  # Set the model to evaluation mode.
    preds = []
    labels = []
    for batch in tqdm(test_loader):
        b_seq, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            pred = model(b_seq.float())
            if label_num == 1:
                preds.extend(pred.squeeze().tolist())
            else:
                softmax = nn.Softmax()
                # preds.extend(softmax(pred)[:,0].tolist())
                preds.extend(pred[:,0].tolist())
            labels.extend(b_labels.int().detach().cpu())
    preds = torch.tensor(preds)
    labels = torch.tensor(labels)
    return preds, labels

class Eval(object):
    """
    Class for evaluation methods
    """
    @staticmethod
    def metrics(metrictypes):
        metrics_dict = {
            "r2": r2_score,
            "mse": mean_squared_error,
            "pcc": pearsonr,
            "scc": spearmanr,
            "auroc": roc_auc_score,
            "auprc": average_precision_score,
            "cls": classification_report,
            "mcc": matthews_corrcoef
        }
        return {metrictype: metrics_dict[metrictype] for metrictype in metrictypes if metrictype in metrics_dict}

    @staticmethod
    def calculate_scores(pred, label, metrictypes,cls_threshold):
        pred = np.array(pred).reshape(-1)
        label = np.array(label).reshape(-1)

        metric_funcs = Eval.metrics(metrictypes)
        scores = {}

        for metrictype, func in metric_funcs.items():
            try:
                if metrictype in ['pcc', 'scc']:
                    scores[metrictype] = func(label, pred)[0]
                elif metrictype in ['cls','mcc']:
                    label_names = {'0':0, '1':1}
                    print("cls_threshold: ",cls_threshold)
                    cls_pred = np.array(pred >= cls_threshold, dtype=int)
                    if metrictype == 'cls':
                        label_names = {'0':0, '1':1}
                        cls_result = func(label, cls_pred, target_names=label_names)
                        print(cls_result)
                    else: # mcc
                        scores[metrictype] = func(label, cls_pred)
                else: # aucroc, aucpr
                    scores[metrictype] = func(label, pred)
            except:
                scores[metrictype] = {}
        return scores

def pick_threshold(record_epoch):
    threshold_file = f'./threshold_pick_{record_epoch}.txt'
    
    if os.path.exists(threshold_file):
        df = pd.read_csv(threshold_file, sep='\t')
        label0 = df[df['label'] == 0]['prediction']
        label1 = df[df['label'] == 1]['prediction']
        from scipy import stats
        from scipy.optimize import brentq
        kde0 = stats.gaussian_kde(label0)
        kde1 = stats.gaussian_kde(label1)
        intersection_point = brentq(lambda x : kde0(x) - kde1(x), df['prediction'].min(), df['prediction'].max())
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.kdeplot(
        data=df, x="prediction", hue="label",
        fill=True, common_norm=False, palette="crest",
        alpha=.5, linewidth=0,
        )

        plt.title('KDE plot of logits for ESM-1v')
        plt.axvline(intersection_point, color='red')
        plt.text(intersection_point-0.3, 0.05, f'x={intersection_point:.3f}', color='red', ha='right')
        os.makedirs('./result/figure',exist_ok=True)
        plt.savefig(f'./result/figure/KDE_ESM-1v_epoch{record_epoch}.png')
    else:
        intersection_point=None
    
    return intersection_point

def predict_results(y_true, preds, name_list, config):
    intersection_point = pick_threshold(config.record_epoch)
    if intersection_point:
        config.cls_threshold = intersection_point

    scores = Eval.calculate_scores(preds, y_true, config.metrics, config.cls_threshold)
    for metrictype, score in scores.items():
        logging.warning(f"{metrictype}: {score:.4f}\n")
    # Saving the prediction results for each test data

    result = pd.DataFrame({
        'target_id': name_list,
        'label': y_true.tolist(),
        'prediction': preds.tolist()
    })
    result.to_csv(config.save_path_predictions, sep='\t', index=False)

    print(f"Predictions saved to {config.save_path_predictions}")

    scores = {k: float(v) if isinstance(v, np.float32)
              else v for k, v in scores.items()}
    json.dump(scores, open(config.save_path_metrics, "w"), indent=4, sort_keys=True)
    print(f"Metrics saved to {config.save_path_metrics}")
