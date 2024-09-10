source /home4/intern/ycwang42/.bashrc
cd /train20/intern/permanent/ycwang42/binarual_enhancement/prop-compare
CUDA_VISIBLE_DEVICES=0,1,2,3 /home4/intern/ycwang42/anaconda3/envs/iwslta100/bin/python -m torch.distributed.launch --nproc_per_node=4 train.py --save_folder exp/cdf