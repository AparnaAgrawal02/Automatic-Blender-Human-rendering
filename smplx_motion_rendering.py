#& 'C:/Program Files/Blender Foundation/Blender 3.6/blender.exe'  -b ../signlanguage_smplerx.blend --python ./try.py^
import bpy
import os
# from genmotion.render.blender.utils import *
# bpy.data.scenes['Scene'].render.engine = 'CYCLES'
# bpy.data.scenes['Scene'].cycles.device = 'GPU'


def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    devices = cycles_preferences.devices

    if not devices:
        raise RuntimeError("Unsupported device type")

    activated_gpus = []
    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
        else:
            device.use = True
            activated_gpus.append(device.name)
            print('activated gpu', device.name)

    cycles_preferences.compute_device_type = device_type
    bpy.context.scene.cycles.device = "GPU"

    return activated_gpus



enable_gpus("CUDA")
import inspect
import numpy as np
import time

for num in range(100,180):
    strt = time.time()
    #remove old objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

    # "C:\\Users\\aparn\\Desktop\\m_01_alb.002.png"
    file = "/ssd_scratch/cvit/aparna/blender_final/female/smplx_animation/answer/"+str(num)+".npz"
    #data = np.load(r"C:/Users/aparn/OneDrive/Desktop/bedlam_render/blender/smplx_anim_to_alembic/animate/animation_sign.npz", allow_pickle=True)
    bpy.data.window_managers["WinMan"].smplx_tool.smplx_gender = 'female'
    bpy.data.window_managers["WinMan"].smplx_tool.smplx_corrective_poseshapes = True

    #bpy.context.view_layer.objects.active = bpy.data.objects['SMPLX-neutral']
    #bpy.context.space_data.shading.type = 'MATERIAL'

    

    bpy.ops.object.smplx_add_animation(filepath=file,hand_reference='RELAXED' , anim_format='SMPL-X')
    bpy.ops.object.smplx_set_poseshapes()
    bpy.ops.material.new()
    mat = bpy.data.materials['Material']
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",os.path.exists("/home2/aparna/f.png"))
    bpy.ops.image.open(filepath="/home2/aparna/f.png", directory="/home2/aparna/", files=[{"name":"f.png", "name":"f.png"}], show_multiview=False)
    texImage = mat.node_tree.nodes.new('ShaderNodeTexImage')
    texImage.image = bpy.data.images['f.png']
    mat.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
    bpy.context.object.active_material = mat
    


    #data = np.load(r"C:/Users/aparn/OneDrive/Desktop/bedlam_render/blender/smplx_anim_to_alembic/animate/animation_sign.npz", allow_pickle=True)
    bpy.context.scene.render.filepath = '/ssd_scratch/cvit/aparna/blender_final/female/answer/smpl_anim_'+ str(num)+'_'
    data = np.load(file, allow_pickle=True)
    frames = data['trans'].shape[0]
    print(frames)
    bpy.data.scenes['Scene'].frame_end = frames
    bpy.ops.render.render(animation=True, use_viewport=True)
    end = time.time()
    print(end-strt)






