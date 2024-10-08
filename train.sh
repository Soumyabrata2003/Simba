##!/bin/bash

#python main.py  \
#--config config/nturgbd120-cross-subject/joint.yaml --model model.Hyperformer.Model --work-dir work_dir/ntu120/csub/Hyperformer_joint --device 2 3 --base-lr 2.5e-2 

#python main.py  \
#--config config/nturgbd120-cross-set/joint.yaml --model model.Hyperformer.Model --base-lr 2.5e-2 

#python main.py  \
#--config config/nturgbd-cross-view/joint.yaml --model model.Hyperformer.Model --base-lr 2.5e-2

#python main.py  \
#--config config/nturgbd-cross-subject/joint.yaml

#python main.py  \
#--config config/nturgbd-cross-subject/bone.yaml

#python main.py  \
#--config config/kinetics-skeleton/joint.yaml --model model.Hyperformer.Model --work-dir work_dir/kinetics-skeleton/Hyperformer_joint  --device 0 --base-lr 2.5e-2
#python main.py  \
#--config config/kinetics-skeleton/joint.yaml --model model.Hyperformer.Model --device 0 

python main.py  \
--config config/ucla/joint.yaml --model model.Hyperformer.Model --work-dir work_dir/ucla/Hyperformer_joint_9

#python main.py  \
#--config config/ucla/bone.yaml --model model.Hyperformer.Model --work-dir work_dir/ucla/Hyperformer_bone_2

#python main.py  \
#--config config/nturgbd-cross-view/bone_vel.yaml --model model.Hyperformer.Model --base-lr 2.5e-2
