# gcn
python3 main.py dataset.input_type=avg model.model_choice=gcn general.save_num=1
python3 main.py dataset.input_type=avg model.model_choice=gcn model.l2=1e-2 general.save_num=2
python3 main.py dataset.input_type=avg model.model_choice=gcn model.l2=1e-2 model.lrs=true general.save_num=3

python3 main.py dataset.input_type=max model.model_choice=gcn general.save_num=4
python3 main.py dataset.input_type=max model.model_choice=gcn model.l2=1e-2 general.save_num=5
python3 main.py dataset.input_type=max model.model_choice=gcn model.l2=1e-2 model.lrs=true general.save_num=6

python3 main.py dataset.input_type=concat model.model_choice=gcn general.save_num=7
python3 main.py dataset.input_type=concat model.model_choice=gcn model.l2=1e-2 general.save_num=8
python3 main.py dataset.input_type=concat model.model_choice=gcn model.l2=1e-2 model.lrs=true general.save_num=9


python3 main.py dataset.input_type=concat model.model_choice=gcn  model.batch_size=1024 general.save_num=10
python3 main.py dataset.input_type=concat model.model_choice=gcn  model.l2=1e-2 model.batch_size=1024 general.save_num=11
python3 main.py dataset.input_type=concat model.model_choice=gcn  model.l2=1e-2 model.lrs=true model.batch_size=1024 general.save_num=12

python3 main.py dataset.input_type=avg model.model_choice=gcn  model.batch_size=1024 general.save_num=13
python3 main.py dataset.input_type=avg model.model_choice=gcn  model.l2=1e-2 model.batch_size=1024 general.save_num=14
python3 main.py dataset.input_type=avg model.model_choice=gcn  model.l2=1e-2 model.lrs=true model.batch_size=1024 general.save_num=15



#gat
python3 main.py dataset.input_type=avg model.model_choice=gat general.save_num=1
python3 main.py dataset.input_type=avg model.model_choice=gat model.l2=1e-2 general.save_num=2
python3 main.py dataset.input_type=avg model.model_choice=gat model.l2=1e-2 model.lrs=true general.save_num=3

python3 main.py dataset.input_type=max model.model_choice=gat general.save_num=4
python3 main.py dataset.input_type=max model.model_choice=gat model.l2=1e-2 general.save_num=5
python3 main.py dataset.input_type=max model.model_choice=gat model.l2=1e-2 model.lrs=true general.save_num=6

python3 main.py dataset.input_type=concat model.model_choice=gat general.save_num=7
python3 main.py dataset.input_type=concat model.model_choice=gat model.l2=1e-2 general.save_num=8
python3 main.py dataset.input_type=concat model.model_choice=gat model.l2=1e-2 model.lrs=true general.save_num=9

python3 main.py dataset.input_type=concat model.model_choice=gat  model.batch_size=1024 general.save_num=10
python3 main.py dataset.input_type=concat model.model_choice=gat  model.l2=1e-2 model.batch_size=1024 general.save_num=11
python3 main.py dataset.input_type=concat model.model_choice=gat  model.l2=1e-2 model.lrs=true model.batch_size=1024 general.save_num=12

python3 main.py dataset.input_type=avg model.model_choice=gat  model.batch_size=1024 general.save_num=13
python3 main.py dataset.input_type=avg model.model_choice=gat  model.l2=1e-2 model.batch_size=1024 general.save_num=14
python3 main.py dataset.input_type=avg model.model_choice=gat  model.l2=1e-2 model.lrs=true model.batch_size=1024 general.save_num=15