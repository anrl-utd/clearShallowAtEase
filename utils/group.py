import os
import cv2
import numpy as np
from PIL import Image

def group_frames(dir_):
    frame_list = []
    for f in os.listdir(dir_):
        print(f)
        frame = int(f[-7:-4])
        frame_list.append(frame)
    print(frame_list)
    grouped_frames = [[] for x in range(243)]
    for frame in frame_list:
        for f in os.listdir(dir_):
            if '.jpg' not in f:
                f += '.jpg'
            num = f[-7:-4]
            zeros = 3 - len(str(frame))
            frame_zeros = ('0'*zeros) + str(frame)
            if str(frame_zeros) == num:
                if dir_ + f not in grouped_frames[frame]:
                    grouped_frames[frame].append(dir_ + f)

    # grouped frames is [frame][list of files that belong to that frame]
    for i in range(243):
        # check if the frame is in the dir we read from
        if not grouped_frames[i]:
            print('Frame ' + str(i) + ' not in dir')
            continue

        print(grouped_frames[i])
        angles = grouped_frames[i]
        imgs_per_cam = [0 for x in range(6)]
        
        for x in range(len(angles)):
            camera = angles[x][-16:-15]
            print(camera)
            imgs_per_cam[int(camera)] += 1
        max_imgs = max(imgs_per_cam)

        print(max_imgs)
        for j in range(len(imgs_per_cam)):
            while imgs_per_cam[j] < max_imgs:
                cam_num = imgs_per_cam[j]
                blank_image = np.zeros((32, 32, 3), np.uint8)
                zeros = 8 - len(str(i))
                save_name = dir_ + 'c' + str(j) + '_' + str(cam_num) + '_' + ('0'*zeros) + str(i) + '.jpg'
                print(save_name)
                print(imgs_per_cam[j])
                cv2.imwrite(save_name, blank_image)
                imgs_per_cam[j] += 1
        print(imgs_per_cam)
