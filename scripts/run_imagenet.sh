# set gpus
export CUDA_VISIBLE_DEVICES=0
nproc_per_node=1
sparsity_type="mix"
weight_decay=8e-5
batchsize=128
train_batch_size=`expr $batchsize / $nproc_per_node`
test_batch_size=`expr $batchsize / $nproc_per_node`
echo $train_batch_size
echo $test_batch_size
model="resnet50"
pretrained="./pretrained/${model}.pth"
loss="CE"
budget=0.45
lower=0.2
mix_version=10
save_dir="./imagenet_exp/${model}_${budget}_${batchsize}_${weight_decay}_${loss}_mix_version_${mix_version}"

train_data_path=""
val_data_path=""

python -m torch.distributed.launch --nproc_per_node=$nproc_per_node --master_port 29501 \
main_imagenet.py imagenet --model $model --save_dir $save_dir \
--budget $budget --sparsity_type $sparsity_type \
--train_batch_size $train_batch_size  \
--test_batch_size $test_batch_size  \
--weight_decay $weight_decay --use_ca \
--pretrained $pretrained --mix_number ${mix_version} \
--train_data_path $train_data_path \
--val_data_path $val_data_path

#python -m torch.distributed.launch --nproc_per_node=1 --master_port 29501 main_imagenet.py imagenet --model resnet50 --save_dir ./imagenet_exp/xxx --budget 0.45 --sparsity_type mix --train_batch_size 128  --test_batch_size 128  --weight_decay 8e-5 --use_ca --pretrained ./pretrained/resnet50 --mix_number 10 --train_data_path xxx --val_data_path xxx