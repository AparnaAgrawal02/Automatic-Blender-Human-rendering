# human_rendering

SMPL-X add on zip with Load aniamtion supporting expression:  
 https://drive.google.com/file/d/1RmnKvDTsBw_5fUfsXzRjXxEA4Izyblsa/view?usp=sharing

original: https://gitlab.tuebingen.mpg.de/jtesch/smplx_blender_addon

## To Run
 CUDA_VISIBLE_DEVICES=1 blender-3.3.17-linux-x64/blender -b smplx_anim_render3.3.blend  -E CYCLES --python blender_final/smplx_motion_rendering.py   -- --cycles-device CUDA
