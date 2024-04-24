import numpy as np
import os
from interpolation import slerp_poses,smooth_poses
import torch
import random
import math
import cv2 
import mediapipe as mp 
from google.protobuf.json_format import MessageToDict 
n_frames = 10
path = '/scratch/aparna/demo/ASL/results/'
image_path = '/scratch/aparna/demo/ASL/images/'
with open("gloss/ASL/asl_ques_gloss_text.txt") as f:
    answers = f.readlines()
y = 0

mpHands = mp.solutions.hands 
hands = mpHands.Hands( 
    static_image_mode=False, 
    model_complexity=1, 
    min_detection_confidence=0.75, 
    min_tracking_confidence=0.75, 
    max_num_hands=2) 
    
for answer in answers:
    print(answer)
    answer = answer[:-1]
    answer = answer.split(" ")
    folders = []
    image_folders = []
    animation = {'motion_info':[], 'betas':[], 'poses':[], 'global_ori':[], 'trans':[], 'mocap_frame_rate':30,'gender':"male","expression":[]}
# ##['motion_info', 'betas', 'poses', 'global_ori', 'trans', 'mocap_frame_rate', 'gender']
    i=0
    if os.path.exists('smplx_animation/'+"question/"+  str(y)+'.npz'):
        y+=1
        continue
    print(len(answer))
    for x in answer:

        #remove space
        x = x.replace(" ","")
        if x == " " or x == "":
            continue
        path_x = os.path.join(path,x)
        image_path_x = os.path.join(image_path,x)
        if os.path.exists(path_x):
            m = os.listdir(path_x)
            #sample 1 file
            file = random.choice(m)
            smplx_folder = os.path.join(path_x, file, 'smplx')
            word_image_folder = os.path.join(image_path_x, file)
            image_folders.append(word_image_folder)
            print(smplx_folder)
            folders.append(smplx_folder)
        else:
            print("Folder does not exist")
    

    for folder,image_folder in zip(folders,image_folders):
        f =1 
        files = os.listdir(folder)
        image_files = os.listdir(image_folder)
        image_files.sort()
        files.sort()
        for file,image_file in zip(files,image_files):
            if file.endswith('.npz'):
                data = np.load(os.path.join(folder, file))
                img = cv2.imread(os.path.join(image_folder, image_file))
                results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                r = data["right_hand_pose"]
            #     fig1 = data["right_hand_pose"][:3].reshape(1,3,3)
            #     fig2 = data["right_hand_pose"][3:6].reshape(1,3,3)
            #     fig3 = data["right_hand_pose"][6:9].reshape(1,3,3)
            #     fig4 = data["right_hand_pose"][9:12].reshape(1,3,3)
            #     fig5 = data["right_hand_pose"][12:15].reshape(1,3,3)
            #     #start from lwer  n ,2->3,3->1, 5->5
            #     #fig5[2] = fig5[2] + np.pi/2
            #     #fig1 ,fig2, fig3, fig4, fig5 = fig3, fig4, fig5, fig2, fig1
            #     #fig1 ,fig2, fig3, fig4, fig5 = fig3, fig5, fig1, fig4, fig2
            #    # r1 = np.vstack((fig1,fig2,fig5,fig4,fig3))
            #     #print(r1.shape,"r1")

                #r = np.vstack((fig1,fig2,fig3,fig4,fig5))
                l = data["left_hand_pose"]
            #     fig1 = data["left_hand_pose"][:3].reshape(1,3,3)
            #     fig2 = data["left_hand_pose"][3:6].reshape(1,3,3)
            #     fig3 = data["left_hand_pose"][6:9].reshape(1,3,3)
            #     fig4 = data["left_hand_pose"][9:12].reshape(1,3,3)
            #     fig5 = data["left_hand_pose"][12:15].reshape(1,3,3)
            #   #  l1= np.vstack((fig1,fig2,fig5,fig4,fig3))
                #start from lwer  1->4 ,2->3,3->1, 5->5
                #fig5[2] = fig5[2] + np.pi/2
                #fig1 ,fig2, fig3, fig4, fig5 = fig3, fig4, fig5, fig2, fig1
                #fig1 ,fig2, fig3, fig4, fig5 = fig3, fig5, fig1, fig4, fig2
                #l = np.vstack((fig1,fig2,fig3,fig4,fig5))
                #l[:,]
                #l  = data["left_hand_pose"]
                # l[:,2] = l[:,2] + np.pi
                # r[:,2] = r[:,2] + np.pi
                #flip x y  
            # l[:,0] , l[:,1] = l[:,1], l[:,0]
            #  r[:,1] , r[:,0] = r[:,0], r[:,1]
             #   r1[:,0,:] , r1[:,1,:],r1[:,2,:] = r1[:,2,:], r1[:,1,:],r1[:,0,:]
             #   l1[:,0,:] , l1[:,1,:],l1[:,2,:] = l1[:,2,:], l1[:,1,:],l1[:,0,:]
               # r1 = r1.reshape(15,-1)
              #  l1 = l1.reshape(15,-1)
                if not results.multi_hand_landmarks:
                    l = np.zeros((15,3))
                    r = np.zeros((15,3))
                if results.multi_hand_landmarks == 1:
                    label = MessageToDict(0)['classification'][0]['label'] 
                    if label == 'right':
                        l = np.zeros((15,3))
                    else:
                        r = np.zeros((15,3))
                pose = np.vstack((data["global_orient"],data["body_pose"], data["jaw_pose"], data["reye_pose"],data["leye_pose"],l,r))
                #print(pose.shape,"pose")
                #print(data["left_hand_pose"].shape,"left_hand_pose")
                #print(data["right_hand_pose"].shape,"right_hand_pose")
                #print(data["jaw_pose"].shape,"jaw_pose")
                #print(data["leye_pose"].shape,"leye_pose")
                # print(data["reye_pose"].shape,"reye_pose")
                # print(data["body_pose"].shape,"body_pose")
                
                
                # print(animation["poses"].shape,"poses")
                if len(animation['betas']) == 0:
                    animation["betas"] = data["betas"]
                else:
                    animation["betas"] = np.vstack((animation["betas"], data["betas"])) 
                    
                    if i != 0 and f == 1:
                        for _ in range(n_frames):
                            animation["betas"] = np.vstack((animation["betas"], data["betas"]))
            
                #pose = pose.reshape(1,-1)
                #normalise
                #pose = pose/np.pi
                #scale
                # #moving average
                # animation["poses"] = np.array(animation["poses"])
                
                # breakpoint()
                if len(animation["poses"]) == 0:
                    animation["poses"] = [pose]
                    #smooth = OneEuroFilter(2, pose)
                else:
                    # animation["poses"] = np.vstack((animation["poses"], pose.reshape(1,-1)))
                    #moving average with window
                    #window = 4 
                   # alpha = 0.5 
                    #pose =  alpha * pose + (1-alpha) * animation["poses"][-1]
                    pose = smooth_poses(torch.tensor(animation["poses"][-1]),torch.tensor(pose),0.65,"rotvec").numpy()
                    #pose = smooth(i,pose)
                  
                    if i != 0 and f == 1:
                        start = animation["poses"][-1]
                        start = torch.tensor(start)
                        end = pose
                        end = torch.tensor(end)
                
                        animation["poses"].extend(slerp_poses(start, end, n_frames,'rotvec').numpy())
                    animation["poses"].append(pose)
                if len(animation["global_ori"]) == 0:
                    #animation["global_ori"] = data["global_orient"].reshape(1,-1)
                    animation["global_ori"] = np.zeros((1,3))
                    # animation["global_ori"][0,2] = 43
                else:
                    
                    animation["global_ori"] = np.vstack((animation["global_ori"], pose[0]))
                 
                    if i != 0 and f == 1:
                        for _ in range(n_frames):
                            animation["global_ori"] = np.vstack((animation["global_ori"], pose[0]))
                    # animation["global_ori"] = np.vstack((animation["global_ori"], np.zeros((1,3))))
                    # animation["global_ori"][-1,2] = 43

                if len(animation["trans"]) == 0:
                    # animation['trans'] = data['transl'].reshape(1,-1)
                    animation['trans'] = np.array([[0,0,0]]).reshape(1,-1)
                else:
                    # animation['trans'] = np.vstack((animation["trans"],  data['transl'].reshape(1,-1)))
                    animation['trans'] = np.vstack((animation["trans"],   np.array([[0,0,0]]).reshape(1,-1)))
                
                    if i != 0 and f == 1:
                        for _ in range(n_frames):   
                            animation["trans"] =np.vstack((animation["trans"],   np.array([[0,0,0]]).reshape(1,-1)))
                if len(animation["expression"]) == 0:
                    animation["expression"] = data["expression"]
                else:
                    animation["expression"] = np.vstack((animation["expression"], data["expression"]))
               
                    if i != 0 and f == 1:
                        for _ in range(n_frames):
                            animation["expression"] = np.vstack((animation["expression"], data["expression"]))

                f = -1
        i+=1
    animation["betas"] = np.mean(animation["betas"], axis=0)
    motion_info = np.array(['turn', '0', '100.0', '0', '377\n'])

    #convert to numpy
    animation["motion_info"] = np.array(animation["motion_info"])
    animation["betas"] = np.array(animation["betas"])
    animation["poses"] = np.array(animation["poses"])
    animation["global_ori"] = np.array(animation["global_ori"])
    print(len(animation["global_ori"]))
    animation['trans'] = np.array(animation["trans"])
    animation["expression"] = np.array(animation["expression"])
    #print shapes
    # print(animation["motion_info"].shape)
    # print(animation["betas"].shape)
    # print(animation["poses"].shape)
    # print(animation["global_ori"].shape)
    # print(animation['trans'].shape)     
    np.savez('smplx_animation/'+"question/"+  str(y)+'.npz', **animation) 
    print("saved,",y)
    y+=1
##['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'betas', 'expression', 'transl']


                   
# please_folder = 'C:/Users/aparn/OneDrive/Desktop/results/69434/smplx'
# drink_folder = 'C:/Users/aparn/OneDrive/Desktop/results/17732/smplx'
# water_folder = 'C:/Users/aparn/OneDrive/Desktop/results/62504/smplx'
# folder_list = [please_folder, drink_folder, water_folder]
# please_files = os.listdir(please_folder)
# please_files = please_files[5:-10]
# drink_files = os.listdir(drink_folder)
# water_files = os.listdir(water_folder)

# n_frames = 10
# animation = {'motion_info':[], 'betas':[], 'poses':[], 'global_ori':[], 'trans':[], 'mocap_frame_rate':30,'gender':"female","expression":[]}
# ##['motion_info', 'betas', 'poses', 'global_ori', 'trans', 'mocap_frame_rate', 'gender']
# please_files.sort()
# drink_files.sort()
# drink_files = drink_files[:-10]
# water_files.sort()
# files_list = [please_files, drink_files, water_files]
# print(len(files_list),"afdfd")
# prev = 0
# for i in range(len(files_list)):
#     print(files_list)
#     f = 1
#     for file in files_list[i]:
   
#         if file.endswith('.npz'):
#             data = np.load(os.path.join(folder_list[i], file))
#             print(len(folder_list))
#             print(folder_list[i])
#             print(os.path.join(folder_list[i], file))
#             print(list(data.keys()  ))
#              # animation['motion_info'].append(data['motion_info'])
#             r = data["right_hand_pose"]
#             fig1 = data["right_hand_pose"][:3].reshape(1,3,3)
#             fig2 = data["right_hand_pose"][3:6].reshape(1,3,3)
#             fig3 = data["right_hand_pose"][6:9].reshape(1,3,3)
#             fig4 = data["right_hand_pose"][9:12].reshape(1,3,3)
#             fig5 = data["right_hand_pose"][12:15].reshape(1,3,3)
#             #start from lwer  1->4 ,2->3,3->1, 5->5
#             #fig5[2] = fig5[2] + np.pi/2
#             #fig1 ,fig2, fig3, fig4, fig5 = fig3, fig4, fig5, fig2, fig1
#             #fig1 ,fig2, fig3, fig4, fig5 = fig3, fig5, fig1, fig4, fig2
#             r1 = np.vstack((fig1,fig2,fig5,fig4,fig3))
#             print(r1.shape,"r1")

#             #r = np.vstack((fig1,fig2,fig3,fig4,fig5))
#             fig1 = data["left_hand_pose"][:3].reshape(1,3,3)
#             fig2 = data["left_hand_pose"][3:6].reshape(1,3,3)
#             fig3 = data["left_hand_pose"][6:9].reshape(1,3,3)
#             fig4 = data["left_hand_pose"][9:12].reshape(1,3,3)
#             fig5 = data["left_hand_pose"][12:15].reshape(1,3,3)
#             l1= np.vstack((fig1,fig2,fig5,fig4,fig3))
#             #start from lwer  1->4 ,2->3,3->1, 5->5
#             #fig5[2] = fig5[2] + np.pi/2
#             #fig1 ,fig2, fig3, fig4, fig5 = fig3, fig4, fig5, fig2, fig1
#             #fig1 ,fig2, fig3, fig4, fig5 = fig3, fig5, fig1, fig4, fig2
#             #l = np.vstack((fig1,fig2,fig3,fig4,fig5))
#             #l[:,]
#             #l  = data["left_hand_pose"]
#             # l[:,2] = l[:,2] + np.pi
#             # r[:,2] = r[:,2] + np.pi
#             #flip x y  
#         # l[:,0] , l[:,1] = l[:,1], l[:,0]
#         #  r[:,1] , r[:,0] = r[:,0], r[:,1]
#             r1[:,0,:] , r1[:,1,:],r1[:,2,:] = r1[:,2,:], r1[:,1,:],r1[:,0,:]
#             l1[:,0,:] , l1[:,1,:],l1[:,2,:] = l1[:,2,:], l1[:,1,:],l1[:,0,:]
#             r1 = r1.reshape(15,-1)
#             l1 = l1.reshape(15,-1)
#             pose = np.vstack((data["global_orient"],data["body_pose"], data["jaw_pose"], data["reye_pose"],data["leye_pose"],r1,l1))
#             print(pose.shape,"pose")
#             print(data["left_hand_pose"].shape,"left_hand_pose")
#             print(data["right_hand_pose"].shape,"right_hand_pose")
#             print(data["jaw_pose"].shape,"jaw_pose")
#             print(data["leye_pose"].shape,"leye_pose")
#             print(data["reye_pose"].shape,"reye_pose")
#             print(data["body_pose"].shape,"body_pose")
            
#             if i != 0 and f == 1:
#                 start = animation["poses"][-1]
#                 start = torch.tensor(start)
#                 end = pose
#                 end = torch.tensor(end)
          
#                 animation["poses"].extend(slerp_poses(start, end, n_frames,'rotvec').numpy())
#                # print(animation["poses"].shape,"poses")
#             if len(animation['betas']) == 0:
#                 animation["betas"] = data["betas"]
#             else:
#                 animation["betas"] = np.vstack((animation["betas"], data["betas"])) 
#                 if i != 0 and f == 1:
#                     for _ in range(n_frames):
#                         animation["betas"] = np.vstack((animation["betas"], data["betas"]))
           
#             #pose = pose.reshape(1,-1)
#             #normalise
#             #pose = pose/np.pi
#             #scale
#             # #moving average
#             # animation["poses"] = np.array(animation["poses"])
            
#             # breakpoint()
#             if len(animation["poses"]) == 0:
#                 animation["poses"] = [pose]
#             else:
#                 # animation["poses"] = np.vstack((animation["poses"], pose.reshape(1,-1)))
                
#                 pose =   animation["poses"][-1] + (pose - animation["poses"][-1])/2
#                 animation["poses"].append(pose)
            
#             if len(animation["global_ori"]) == 0:
#                 #animation["global_ori"] = data["global_orient"].reshape(1,-1)
#                 animation["global_ori"] = np.zeros((1,3))
#                 # animation["global_ori"][0,2] = 43
#             else:
                
#                 animation["global_ori"] = np.vstack((animation["global_ori"], data["global_orient"].reshape(1,-1)))
#                 if i != 0 and f == 1:
#                     for _ in range(n_frames):
#                         animation["global_ori"] = np.vstack((animation["global_ori"], data["global_orient"].reshape(1,-1)))
#                 # animation["global_ori"] = np.vstack((animation["global_ori"], np.zeros((1,3))))
#                 # animation["global_ori"][-1,2] = 43

#             if len(animation["trans"]) == 0:
#                 # animation['trans'] = data['transl'].reshape(1,-1)
#                 animation['trans'] = np.array([[0,0,0]]).reshape(1,-1)
#             else:
#                 # animation['trans'] = np.vstack((animation["trans"],  data['transl'].reshape(1,-1)))
#                 animation['trans'] = np.vstack((animation["trans"],   np.array([[0,0,0]]).reshape(1,-1)))
#                 if i != 0 and f == 1:
#                     for _ in range(n_frames):   
#                         animation["trans"] =np.vstack((animation["trans"],   np.array([[0,0,0]]).reshape(1,-1)))
#             if len(animation["expression"]) == 0:
#                 animation["expression"] = data["expression"]
#             else:
#                 animation["expression"] = np.vstack((animation["expression"], data["expression"]))
#                 if i != 0 and f == 1:
#                     for _ in range(n_frames):
#                         animation["expression"] = np.vstack((animation["expression"], data["expression"]))

#             f = -1
# #print(animation["poses"][0].shape,"poses1")

           
# #take mean of betas
# animation["betas"] = np.mean(animation["betas"], axis=0)
# motion_info = np.array(['turn', '0', '/ps/project/datasets/AMASS/amass_march_2022/mosh_results/smplx_neutral/KIT/314/bend_left01_stageii.npz', '100.0', '0', '377\n'])

# #convert to numpy
# animation["motion_info"] = np.array(animation["motion_info"])
# animation["betas"] = np.array(animation["betas"])
# animation["poses"] = np.array(animation["poses"])
# animation["global_ori"] = np.array(animation["global_ori"])
# animation['trans'] = np.array(animation["trans"])
# animation["expression"] = np.array(animation["expression"])
# #print shapes
# print(animation["motion_info"].shape)
# print(animation["betas"].shape)
# print(animation["poses"].shape)
# print(animation["global_ori"].shape)
# print(animation['trans'].shape)     
# np.savez('animation_please_drink_water.npz', **animation) 
# ##['global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'betas', 'expression', 'transl']



