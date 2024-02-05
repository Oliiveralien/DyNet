model="resnet32"
weight_decay=1e-4
batchsize=64
budget=0.4
sparsity_type="mix"
data_path="./data/cifar10"


########################different alpha#############################

alpha=1
save_dir="./exp/alpha/${model}_${budget}_${alpha}"
python main_cifar.py cifar --model $model --save_dir $save_dir --budget $budget \
     --sparsity_type $sparsity_type  --batchsize $batchsize --weight_decay $weight_decay --loss_alpha $alpha --data_path $data_path

alpha=5
save_dir="./exp/alpha/${model}_${budget}_${alpha}"
python main_cifar.py cifar --model $model --save_dir $save_dir --budget $budget \
     --sparsity_type $sparsity_type  --batchsize $batchsize --weight_decay $weight_decay --loss_alpha $alpha --data_path $data_path

alpha=15
save_dir="./exp/alpha/${model}_${budget}_${alpha}"
python main_cifar.py cifar --model $model --save_dir $save_dir --budget $budget \
     --sparsity_type $sparsity_type  --batchsize $batchsize --weight_decay $weight_decay --loss_alpha $alpha --data_path $data_path

alpha=20
save_dir="./exp/alpha/${model}_${budget}_${alpha}"
python main_cifar.py cifar --model $model --save_dir $save_dir --budget $budget \
     --sparsity_type $sparsity_type  --batchsize $batchsize --weight_decay $weight_decay --loss_alpha $alpha --data_path $data_path

alpha=100
save_dir="./exp/alpha/${model}_${budget}_${alpha}"
python main_cifar.py cifar --model $model --save_dir $save_dir --budget $budget \
     --sparsity_type $sparsity_type  --batchsize $batchsize --weight_decay $weight_decay --loss_alpha $alpha --data_path $data_path

########################different gumbel temp#############################

gumbel_temp=0.1
save_dir="./exp/gumbel_temp/${model}_${budget}_${gumbel_temp}"
python main_cifar.py cifar --model $model --save_dir $save_dir --budget $budget \
     --sparsity_type $sparsity_type  --batchsize $batchsize --weight_decay $weight_decay --gumbel_temp gumbel_temp --data_path $data_path

gumbel_temp=0.5
save_dir="./exp/gumbel_temp/${model}_${budget}_${gumbel_temp}"
python main_cifar.py cifar --model $model --save_dir $save_dir --budget $budget \
     --sparsity_type $sparsity_type  --batchsize $batchsize --weight_decay $weight_decay --gumbel_temp gumbel_temp --data_path $data_path

gumbel_temp=5
save_dir="./exp/gumbel_temp/${model}_${budget}_${gumbel_temp}"
python main_cifar.py cifar --model $model --save_dir $save_dir --budget $budget \
     --sparsity_type $sparsity_type  --batchsize $batchsize --weight_decay $weight_decay --gumbel_temp gumbel_temp --data_path $data_path

gumbel_temp=10
save_dir="./exp/gumbel_temp/${model}_${budget}_${gumbel_temp}"
python main_cifar.py cifar --model $model --save_dir $save_dir --budget $budget \
     --sparsity_type $sparsity_type  --batchsize $batchsize --weight_decay $weight_decay --gumbel_temp gumbel_temp --data_path $data_path



########################different loss#############################

save_dir="./exp/loss_func/${model}_${budget}_mix1"
python main_cifar.py cifar --model $model --save_dir $save_dir --budget $budget \
     --sparsity_type $sparsity_type  --batchsize $batchsize --weight_decay $weight_decay --mix_number 1 --data_path $data_path

save_dir="./exp/loss_func/${model}_${budget}_mix7"
python main_cifar.py cifar --model $model --save_dir $save_dir --budget $budget \
     --sparsity_type $sparsity_type  --batchsize $batchsize --weight_decay $weight_decay --mix_number 7 --data_path $data_path