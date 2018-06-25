import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
# from https://github.com/kcct-fujimotolab/3DCNN/blob/master/videoto3d.py
def video3d(filename, vid_size, color, skip,rand,optical_flow,proj):
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
    if optical_flow:


        ret, old_frame = cap.read()
        # x = np.linspace(-55,55)
        # y = np.linspace(-55,55)
        # XX,YY = np.meshgrid(x,y)
        old_frame = cv2.resize(old_frame, (height, width))
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        old_gray = old_gray.astype(np.float32)
    framearray = []
    pp = []
    for i in range(read):

        cap.set(cv2.CAP_PROP_POS_FRAMES, frames[i])
        ret, frame = cap.read()
        # import pdb; pdb.set_trace()
        frame = frame[:,:450,:]
        frame = cv2.resize(frame, (height, width))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if optical_flow:
            frame_gray = frame_gray.astype(np.uint8)
            old_gray = old_gray.astype(np.uint8)
            flow = cv2.calcOpticalFlowFarneback(old_gray,frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            fin_dummy = np.dot(proj, flow.flatten())
            pp.append(fin_dummy)
            old_gray = frame_gray.copy()
            # p0 = good_new.reshape(-1,1,2)
        if color:
            frame = np.reshape(frame,(3,height,width))
            # import pdb; pdb.set_trace()
            framearray.append(frame)
        else:
            frame = frame_gray
            frame = np.reshape(frame,(1,height,width))
            framearray.append(frame)
    framearray = np.array(framearray)
    framearray = np.reshape(framearray,(-1,depth,height,width))
        # import pdb; pdb.set_trace()
        # framearray = np.reshape()
    cap.release()
    if optical_flow:
        return np.array(framearray),pp
    else:
        return np.array(framearray)



# frame_array = video3d('data/subject_020_posed_smile_2_640x360_30.mp4', (220,220,10), color=False, skip=True)
