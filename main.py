import utils
import pandas as pd
import re
import os
import sys
from collections import Counter
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch
import wandb
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


class PETaseTrain(object):
    """
    PETase Classifier Training
    """

    def __init__(self, cfg: HydraConfig):
        self.seed = cfg.general.seed
        self.gpu_id = cfg.general.gpu_id
        self.emb_path = cfg.dataset.emb_path
        self.dataset = os.path.join(
            self.emb_path, cfg.dataset.datafile) if cfg.dataset.datafile else None

        """ model parameters """
        self.batch_size = cfg.model.batch_size
        self.model_save_path = cfg.model.model_save_path
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)

        """ special parameters """
        self.save_path_log = cfg.general.save_path_log
        self.logits = cfg.general.logits
        self.input_type = cfg.general.input_type
        self.label_num = cfg.model.label_num
        self.loss_fn = cfg.model.loss_fn

        self.define_logging()
        self.load_device()

        if cfg.general.usage != 'feat_extract':
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
                record_id: protein id or protein name as the annotation 
                mt_seq: mutant sequence
                mt_aa_list: mutated amino acids (list)
                wt_aa_list: wild-type amino acids (list)
                aa_index: mutation position index (list)
                label: labels for each data

            *** wt_seq has been set as "{config.wt_seq}" ***
        """

        csv = os.path.join(config.ds_path, config.csv)
        df = pd.read_csv(csv)
        column_names = ['record_id','mt_seq', 'mt_aa_list', 'wt_aa_list', 'aa_index']
        if set(column_names) - set(df.columns): 
            raise ValueError(f'''
                             Your dataframe lacks of "{', '.join(set(column_names) - set(df.columns))}" among columns
                             
                             DataFrame column name instruction for feature extraction:
                             {warning}
                             ''')

        if 'label' not in df.columns:
            print('**** The input dataframe does not contain a "label" column, \
                  here we give "-1" for all data as label to ensure the programme can run properly. ***')
            df['label'] = -1

        logging.warning(f'getting embeds for {csv}')
        logging.warning(f'data length: {df.shape[0]}')

        for model in config.model_list:
            logging.info(f'generating embeds from model: {model}')
            save_embed = os.path.join(self.emb_path, f'{model}.pt')
            os.makedirs(os.path.dirname(save_embed), exist_ok=True)
            utils.generate_embeds_and_save(df, 
                                           wt_seq=config.wt_seq,
                                           save_path=save_embed,
                                            model_selection=model,
                                            device=self.device,
                                            batch_size=config.batch_size_emb_gen)

            logging.info(
                f'Done! Embeds for {csv} (from {model}) has been saved as {save_embed}')

    def load_data(self, usage):
        '''
        input: 
            embeddings file path of dataset (e.g. data/embeds/esm1v.pt)
        '''

        logging.warning(f"Loading data for the purpose of ** {usage} ** ...")

        if self.dataset is None:
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
            logging.warning(f'Loading dataset from {self.dataset}...')
            
        X, y, pid_list = utils.unpickler(ds_name=self.dataset, logits=self.logits, input_type=self.input_type)
        
        if usage == 'train':
            # Calculate weight to solve data imbalance problem
            counter_dict = Counter(y)
            class_counts = [y.count(i) for i in range(self.label_num)]
            min_class_count = min(class_counts)
            if self.label_num == 1:
                self.weights = None
            elif self.label_num == 2:
                weights = counter_dict[0]/counter_dict[1] if counter_dict[0] > counter_dict[1] else counter_dict[1]/counter_dict[0]
                # if pos data amount is larger than that of neg data amount, the value of weights should be large, otherwise small
                # large weights value indicates the model should pay more attention to the pos data
                self.weights = torch.tensor(weights)
            else:
                weights = [1.0 / (count / min(class_counts)) for count in class_counts]
                self.weights = torch.tensor(weights)


            if self.loss_fn != "mse":
                # Calculate weight to solve data imbalance problem
                counter_dict = Counter(y)
                class_counts = [y.count(i) for i in range(self.label_num)]
                min_class_count = min(class_counts)

                if self.loss_fn == "bce":
                    weights = counter_dict[0] / \
                        counter_dict[1] if counter_dict[0] > counter_dict[1] else counter_dict[1]/counter_dict[0]
                    self.weights = torch.tensor(weights)
                else:
                    weights = [1.0 / (count / min(class_counts))
                               for count in class_counts]
                    self.weights = torch.tensor(weights)
            else:
                self.weights = None

            # split X and y
            X_train, X_test, y_train, y_test, pid_train_list, pid_test_list = train_test_split(
                X, y, pid_list, test_size=0.3, shuffle=True,
                stratify=y, random_state=self.seed)

            # split pid_list using the same random state for consistency
            X_train, X_valid, y_train, y_valid, = train_test_split(
                X_train, y_train,
                test_size=0.3, shuffle=True,
                stratify=y_train, random_state=self.seed)

            self.pid_list = pid_test_list
            self.model_dim = X_train.shape[-1]
            logging.warning(f'X_train shape: {X_train.shape}')
            logging.warning(f'X_test shape: {X_test.shape}')
            logging.warning(f'X_valid shape: {X_valid.shape}\n')

            train_dataset = utils.MLPDataset(X_train, y_train)
            val_dataset = utils.MLPDataset(X_valid, y_valid)
            test_dataset = utils.MLPDataset(X_test, y_test)

            # > Feed dataset to dataloader
            self.train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True)
            self.val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size)
            self.test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size)

        elif usage == 'infer':
            logging.warning(f'dataset shape: {X.shape}')
            dataset = utils.MLPDataset(X, y)
            self.test_loader = DataLoader(dataset, batch_size=self.batch_size)
            self.pid_list = pid_list
            self.model_dim = X.shape[-1]
        else:
            raise ValueError(f'''
                             usage should be one of the following:
                                "train" or "infer"
                             Now it is: {usage}
                             ''')

    def load_model(self, config):
        model_size = self.model_dim
        num_hidden = int(model_size/2)

        logging.warning(f'model_size: {model_size}')
        logging.warning(f'num_hidden: {num_hidden}\n')

        logging.warning("Loading model...")
        self.model = utils.MLPClassifier_LeakyReLu(num_input=model_size,
        num_hidden=num_hidden,
        num_output=config.label_num,
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
                      self.weights,
                      train_cfg,
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

        preds, y_true = utils.predict(
            self.test_loader, self.model, self.label_num, self.device)

        os.makedirs(os.path.dirname(
            config.save_path_predictions), exist_ok=True)
        os.makedirs(os.path.dirname(config.save_path_metrics), exist_ok=True)

        utils.predict_results(y_true, preds,
                              self.pid_list, config
                              )


@hydra.main(version_base=None, config_path="./", config_name="config.yaml")
def main(cfg: HydraConfig) -> None:

    PETase_classifier = PETaseTrain(cfg)

    if cfg.general.usage == 'train':
        PETase_classifier.train(cfg)
        PETase_classifier.evaluate(cfg.general)
    elif cfg.general.usage == 'feat_extract':
        PETase_classifier.feature_extraction(cfg.dataset)
    elif cfg.general.usage == 'infer':
        PETase_classifier.evaluate(cfg.general)


if __name__ == '__main__':

    main()

    print('\n=============== No Bug No Error, Finished!!! ===============')
