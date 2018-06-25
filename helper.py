# def crop_image(frame):
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
# from https://github.com/kcct-fujimotolab/3DCNN/blob/master/videoto3d.py
def video3d(filename, vid_size, color, skip,rand):
    cap = cv2.VideoCapture(filename)
    nframe = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    height, width, depth  = vid_size
    if skip:
#         st = np.random.randint(depth-2)
        if rand:
            st = np.random.randint(math.floor(nframe/depth)-1)
        else:
            st = 0
        frames = [int(x * nframe / depth) + st for x in range(depth)]
        read = depth
    else:
        start = int(nframe/2) - int(depth/2)
        if rand:
            st = np.random.randint(math.floor(nframe-(depth+start)))
        else:
            st = 0
        frames = np.arange(start, start+depth) + st
        read = depth
    framearray = np.empty((0,height, width))
    rnd = np.random.rand()
    ii=0
    while ii < len(frames):
#     for i in range(len(frame)):
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[ii])
        ret, frame = cap.read()
        if ret == False:
            frames = frames + 1
            continue
        else:
            ii = ii+1
        
#         try:
        frame = frame[:,:450,:]
#         except:
#             import pdb; pdb.set_trace()

        frame = cv2.resize(frame, (height, width))
        if rnd < .5:
            frame = cv2.flip(frame, 0 )
        if color:
            frame = np.reshape(frame,(3,height,width))
#             import pdb; pdb.set_trace()
            framearray = np.concatenate((framearray,frame),0)
#             framearray.append(frame)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = np.reshape(frame,(1,height,width))
            framearray = np.concatenate((framearray,frame),0)
#             framearray.append(frame)
            
#     framearray = np.array(framearray)
    framearray = np.reshape(framearray,(-1,depth*frame.shape[0],height,width))
        # import pdb; pdb.set_trace()
        # framearray = np.reshape()
    cap.release()
    return np.array(framearray)

# frame_array = video3d('data/subject_020_posed_smile_2_640x360_30.mp4', (220,220,10), color=False, skip=True)
