# human_rendering
## blend file
https://drive.google.com/file/d/1v-eTzouZrBqMvxnmPYdPlycvbQ1FDd1u/view?usp=sharing
## To Run
 CUDA_VISIBLE_DEVICES=1 blender-3.3.17-linux-x64/blender -b smplx_anim_render3.3.blend  -E CYCLES --python blender_final/smplx_motion_rendering.py   -- --cycles-device CUDA
