#$ -l tmem=2G,h_rt=12:00:0,gpu=true
#$ -N MKK
#$ -S /bin/bash
#$ -P cath
#$ -j y
#$ -R y
#$ -cwd
#$ -t 1-3

hostname
date

source /share/apps/source_files/python/python-3.8.5.source
source /share/apps/source_files/anaconda/conda-2022-5.source
module load python/3

. /home/weinilin/pytorch/bin/activate
which python3
python3 --version

DOM_ID=$(head -n $SGE_TASK_ID /SAN/orengolab/plm_embeds/MKK_classifier/todo | tail -n 1)

# Replace the variables in the config.yaml file based on the paths where you store models and data, or you can specify them directly here
# For example, if you wanna change the learning rate and batch_size, you can do:
# python3 main.py model.lr=0.0001 model.batch_size=16

python3 main.py ${DOM_ID}

# python3 main.py general.usage=feat_extract dataset.model_list=["esm1b"]
# python3 main.py general.usage=feat_extract dataset.model_list=["esm2"]

echo "Done"