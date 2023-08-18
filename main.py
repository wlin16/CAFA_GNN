import utils
import pandas as pd
import re
import os
import sys
import pickle
import logging
from sklearn.model_selection import train_test_split
from dgl.dataloading import GraphDataLoader
import torch
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class GNNTrain(object):
    """
    GNN Training
    """

    def __init__(self, cfg: HydraConfig):
        self.seed = cfg.general.seed
        self.gpu_id = cfg.general.gpu_id
        self.ds_path = cfg.dataset.ds_path
        self.csv = cfg.dataset.csv
        self.graph_path = cfg.dataset.graph_path
        self.go_dict = cfg.dataset.go_dict

        """ model parameters """
        self.batch_size = cfg.model.batch_size
        self.model_save_path = cfg.model.model_save_path
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

        """ special parameters """
        self.save_path_log = cfg.general.save_path_log
        self.input_type = cfg.general.input_type
        self.label_num = cfg.model.label_num
        self.loss_fn = cfg.model.loss_fn

        self.define_logging()
        self.load_device()

        if cfg.general.usage not in ('feat_extract', 'plot'):
            self.load_data(cfg.general.usage)
            self.load_model(cfg.model)

    def define_logging(self):
        # Create a logger
        logging.basicConfig(
            level=logging.DEBUG,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %A %H:%M:%S',
            filename=self.save_path_log,
            filemode='w')
        # Define a Handler and set a format which output to console
        console = logging.StreamHandler()
        console.setLevel(logging.WARNING)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)

    def load_wandb(self, cfg):
        if cfg.wandb.run_id:
            wandb_run_id = cfg.wandb.run_id
        else:
            wandb_run_id = wandb.util.generate_id()
            OmegaConf.set_struct(cfg, False)
            cfg.wandb.run_id = wandb_run_id
            OmegaConf.set_struct(cfg, True)

        self.wandb_run = wandb.init(
            project=cfg.wandb.project,
            config={
                "dropout": cfg.model.dropout,
                "batch_size": cfg.model.batch_size,
                "lr": cfg.model.lr,
                "epochs": cfg.model.n_epochs,
                "seed": cfg.general.seed,
                "early_stop": cfg.model.early_stop,
            },
            resume=True,
            name=cfg.wandb.run_name,
            id=wandb_run_id,
        )
        
        logging.warning(f'wandb_run_id: {wandb_run_id}')
        logging.warning(f'wandb_run_name: {cfg.wandb.run_name}')

    def load_device(self):
        SEED = self.seed
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu_id}")
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            logging.warning(
                f'There are {torch.cuda.device_count()} GPU(s) available.')
            logging.warning(f'Device name: {torch.cuda.get_device_name(0)}')
        else:
            logging.warning("////////////////////////////////////////////////")
            logging.warning("///// NO GPU DETECTED! Falling back to CPU /////")
            logging.warning("////////////////////////////////////////////////")
            self.device = torch.device("cpu")

    def feature_extraction(self, config):
        '''Extract embeddings from given .csv file by specified esm (e.g esm1v, esm1b, esm2)'''

        warning = f"""
        Input: Specify csv file name of 'csv' option under the dataset section in config.yaml.
            Necessary columns as following:
                UniProtID1: target protein name as the annotation 
                UniProtID2: neighbour protein name as the annotation 
                GO_terms_associated_to_UniProtID1: GO term labels for target protein (list)
                GO_terms_associated_to_UniProtID2: GO term labels for neighbour protein (list)
        """

        csv = os.path.join(self.ds_path, self.csv)
        df = pd.read_csv(csv)
        go_dict = pickle.load(open(os.path.join(self.ds_path, self.go_dict), "rb"))
        go_dict_len = len(go_dict)

        if self.input_type != "concat":
            emb = os.path.join(self.ds_path, self.input_type) + ".pt"
            embedding = torch.load(emb)
        else:
            avg_dict = torch.load(self.ds_path + '/' + 'avg.pt')
            max_dict = torch.load(self.ds_path + '/' + 'max.pt')
            embedding = {}
            for key in avg_dict:
                embedding[key] = torch.cat([avg_dict[key], max_dict[key]])

        column_names = ['UniProtID1','UniProtID2', 'GO_terms_associated_to_UniProtID1', 'GO_terms_associated_to_UniProtID2']
        if set(column_names) - set(df.columns): 
            raise ValueError(f'''
                             Your dataframe lacks of "{', '.join(set(column_names) - set(df.columns))}" among columns
                             
                             DataFrame column name instruction for feature extraction:
                             {warning}
                             ''')

        if 'GO_terms_associated_to_UniProtID1' not in df.columns:
            print('**** The input dataframe does not contain a "label" column, \
                  here we give "-1" for all data as label to ensure the programme can run properly. ***')
            df['GO_terms_associated_to_UniProtID1'] = -1

        logging.warning(f'getting embeds for {csv}')
        logging.warning(f'data length: {df.shape[0]}')

        logging.warning(f'generating embeds for: {self.input_type}')
        os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
        utils.build_subgraphs(df, embedding=embedding,
                              go_dict_len=go_dict_len,
                              save_path=self.graph_path)

        logging.warning(
            f'Done! Graphs for {csv} ({self.input_type}) have been saved as {self.graph_path}')

    def load_data(self, usage):
        '''
        input: 
            embeddings file path of dataset (e.g. data/embeds/esm1v.pt)
        '''

        logging.warning(f"Loading data for the purpose of ** {usage} ** ...")

        graph_num = os.path.dirname(self.graph_path)

        if len(graph_num) == 0:
            raise ValueError('''
                             We cannot detect the prepared dataset (e.g. data/embeds/esm1v.pt).
                             Please:
                                1. Specify the dataset file name in config.yaml
                                2. If you have not prepared the dataset:
                                    2.1 Please prepare your csv file containing the protein sequences and the corresponded mutation information
                                    2.2 Specify the csv file name in config.yaml
                                    2.3 Set general.usage as "feat_extract"
                                    2.4 Run the code to generate embeddings for your data. E.g.: 
                                    python3 main.py dataset.csv=./data/mkk.csv general.usage="feat_extract"
                             ''')
        else:
            logging.warning(f'Loading graphs from {self.graph_path}...')
        

        csv = os.path.join(self.ds_path, self.csv)
        df = pd.read_csv(csv)
        self.go_dict = pickle.load(open(os.path.join(self.ds_path, self.go_dict), "rb"))
        self.go_dict_len = len(self.go_dict)

        unique_proteins = df['UniProtID1'].unique().tolist()
        
        if usage == 'train':
            # Calculate weight to solve data imbalance problem
            train,temp = train_test_split(unique_proteins, test_size=0.4, random_state=42)
            valid, test = train_test_split(temp, test_size=0.5, random_state=42)

            logging.warning(f'X_train shape: {len(train)}')
            logging.warning(f'X_test shape: {len(test)}')
            logging.warning(f'X_valid shape: {(len(valid))}\n')

            # Create DataLoader
            self.train_loader = GraphDataLoader(utils.GraphDataset(train,self.graph_path), batch_size=self.batch_size, shuffle=True)
            self.val_loader = GraphDataLoader(utils.GraphDataset(valid,self.graph_path), batch_size=self.batch_size, shuffle=False)
            self.test_loader = GraphDataLoader(utils.GraphDataset(test,self.graph_path), batch_size=self.batch_size, shuffle=False)

            self.pid_list = test

        elif usage == 'infer':
            logging.warning(f'dataset shape: {len(unique_proteins)}')
            dataset = utils.MLPDataset(unique_proteins, self.graph_path)
            self.test_loader = GraphDataLoader(dataset, batch_size=self.batch_size)
            self.pid_list = unique_proteins
            self.model_dim = self.model_dim
        else:
            raise ValueError(f'''
                             usage should be one of the following:
                                "train" or "infer"
                             Now it is: {usage}
                             ''')

    def load_model(self, config):
        model_size = 1024 if self.input_type != 'concat' else 2048
        num_hidden = int(model_size/2)

        logging.warning(f'model_size: {model_size}')
        logging.warning(f'num_hidden: {num_hidden}\n')

        logging.warning("Loading model...")
        if config.model_choice == 'gcn':
            self.model = utils.GCN(num_input=model_size,
            num_hidden=num_hidden,
            num_output=self.go_dict_len,
            dropout=config.dropout
            ).to(self.device)
        elif config.model_choice == 'gat':
            self.model = utils.GAT(num_input=model_size,
            num_hidden=num_hidden,
            num_output=self.go_dict_len,
            num_heads=config.num_heads,
            dropout=config.dropout
            ).to(self.device)
       
        logging.warning("Model loaded.\n")

    def train(self, cfg):
        '''
        output:
            model weights (result/model/model.ckpt)
            valid predictions for picking the best mcc threshold (e.g. ./threshold.txt)
        '''

        total_params = sum(p.numel() for p in self.model.parameters())
        logging.warning(f'total parameters: {total_params:,}')
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.warning(
            f'total training parameters: {total_trainable_params:,}')

        logging.warning("Start training...")

        self.load_wandb(cfg)
        train_cfg = cfg.model
        utils.trainer(self.train_loader,
                      self.val_loader,
                      self.model,
                      train_cfg,
                      self.go_dict_len,
                      self.device)
        wandb.finish()

    def evaluate(self, config):
        '''
        input:
            embeddings file path of your dataset (e.g. data/embeds/esm1v.pt)
        output: 
            cls: model performance (MCC, AUC, classification_report) (e.g. ../example/output_results/model_performance.txt)
            reg: model performance (RMSE, R2, PCC, SCC) (e.g. ../example/output_results/model_performance.txt)
        '''

        logging.warning("Start evaluating...")

        checkpoint = torch.load(self.model_save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        preds, y_true = utils.predict(self.test_loader, self.model, 
                                      self.device, config.usage)

        os.makedirs(os.path.dirname(config.save_path_predictions), exist_ok=True)
        os.makedirs(os.path.dirname(config.save_path_metrics), exist_ok=True)
        
        utils.predict_results(y_true, preds,
                              name_list = self.pid_list,
                            save_path_predictions = config.save_path_predictions,
                            save_path_metrics = config.save_path_metrics,
                            save_num= config.save_num,
                            label_num =self.go_dict_len,
                            go_dict= self.go_dict,
                            device = self.device,
                            usage=config.usage
                            )


@hydra.main(version_base=None, config_path="./", config_name="config.yaml")
def main(cfg: HydraConfig) -> None:

    GNN_classifier = GNNTrain(cfg)

    if cfg.general.usage == 'train':
        GNN_classifier.train(cfg)
        GNN_classifier.evaluate(cfg.general)
    elif cfg.general.usage == 'feat_extract':
        GNN_classifier.feature_extraction(cfg.dataset)
    elif cfg.general.usage == "plot":
        utils.plot_and_save_subgraphs(cfg.dataset.graph_path, cfg.dataset.plot, 
                                          cfg.dataset.plot_out)
    elif cfg.general.usage == 'infer':
        GNN_classifier.evaluate(cfg.general)


if __name__ == '__main__':

    main()

    print('\n=============== No Bug No Error, Finished!!! ===============')
