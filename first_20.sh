#!/bin/bash
#SBATCH -A aparna
#SBATCH -c 10
#SBATCH -w gnode051
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=10-00:00:00
#SBATCH --mail-type=END
#SBATCH --output=render_20.txt

#rsync -r jayasree_saha@preon.iiit.ac.in:/home/rudrabha_mukhopadhyay/SPEAKER_EMBEDDING_VOX2/speaker_embedding /scratch/aparna/ -p cvit@1234 --progress --ignore-existing
#tar -cf /scratch/aparna/VoxCeleb2_dev_test/dev/voceleb2.tar.gz  /scratch/aparna/VoxCeleb2_dev_test/dev/mp4 
#tar -xf /scratch/aparna/vox2_train_val.tar /scratch/aparna/     --checkpoint=.10000  
CUDA_VISIBLE_DEVICES=1 blender-3.3.17-linux-x64/blender -b smplx_anim_render3.3.blend  -E CYCLES --python blender_final/smplx_motion_rendering.py   -- --cycles-device CUDA
