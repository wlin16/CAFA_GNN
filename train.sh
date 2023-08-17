#$ -l tmem=2G,h_rt=12:00:0,gpu=true
#$ -N MKK_MLP_train
#$ -S /bin/bash
#$ -P cath
#$ -j y
#$ -R y
#$ -cwd
#$ -t 1-1

hostname
date

source /share/apps/source_files/python/python-3.8.5.source
source /share/apps/source_files/anaconda/conda-2022-5.source
module load python/3

. /home/weinilin/pytorch/bin/activate
which python3
python3 --version


python3 main.py general.logits=false \
    model.batch_size=128 \
    wandb.run_name=bce_esm1v_bs128 \
