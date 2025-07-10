cd /home/ubuntu/xiegroup/tianmin-project/yyin34/codebase/Video-Depth-Anything

python divide_files.py --root_dir /home/ubuntu/xiegroup/tianmin-project/yyin34/dataset/re10k/train

python3 ./metric_depth/run_metric.py --root_dir /home/ubuntu/xiegroup/tianmin-project/yyin34/dataset/re10k/unit --save_npz --encoder vitl --folders_file /home/ubuntu/xiegroup/tianmin-project/yyin34/codebase/Video-Depth-Anything/gpu_splits/group_0.txt