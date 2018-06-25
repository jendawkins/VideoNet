from sklearn.neighbors import NearestNeighbors
from helper_optflow import *
import matplotlib.pyplot as plt
import random
import pdb
import numpy as np
import math
import re
from sklearn.svm import SVC
from scipy.stats import mode


class Trainer():
    def __init__(self,train_path,test_path,vid_path,timesteps,epochs,
                 color,random_sample, skip,
                 optical_flow):
        self.train_path = train_path
        self.optical_flow = optical_flow
        self.test_path = test_path
        self.vid_path = vid_path
        self.epochs = epochs
        self.timesteps = timesteps
        self.color = color
        self.random_sample = random_sample
        self.skip = skip
        self.rng = np.random.RandomState(123456)
        self.proj = self.rng.randn(64,2*110*110).astype(np.float32)

    # def get_accuracy(self,outputs,labels):
    #     _, predicted = torch.max(outputs.data, 1)
    #     correct = float(((predicted == labels.detach()).sum()).item())
    #     total = labels.size(0)
    #     return correct, total

    def get_data(self, path):
        content_t = []
        if isinstance(path,str):
            with open(path) as f:
                content_t = f.readlines()
        else:
            for pat in path:
                with open(pat) as f:
                    content = f.readlines()
                content_t.extend(content)
        labels = [re.findall('<([^>]*)>', c)[0] for c in content_t]
        x = list(set(labels))
        dic = dict(zip(x, list(range(0,len(x)))))
        num_labels = [dic[v] for v in labels]

        paths = [re.findall('"([^"]*)"', c)[0] for c in content_t]
        return paths, labels


    def train_test(self,train_path,test_path):
        loss_vec = []
        acc_vec = []
        # net = VidNet(self.timesteps,self.method)
        content_t = []
        train_paths, train_labels = self.get_data(train_path)
        test_paths, test_labels = self.get_data(test_path)
        correct = 0
        total = 0
        flpts_array = []
        for vid_name in train_paths:
            frame, flpts = video3d('hollywood/'+self.vid_path + '/' + vid_name, (110,110,self.timesteps),
                                    color=self.color,skip=self.skip,rand=self.random_sample,
                                    optical_flow=self.optical_flow,proj = self.proj)
            # (x_new,y_new),(x_old,y_old)  N x 2, N x 2
            # new = flpts[0][0]

            flpts = np.array(flpts)

            flpts_array.append(flpts.flatten())
            # import pdb; pdb.set_trace()
        flpts_array = np.array(flpts_array)
        flptst_array = []
        for test_name in test_paths:
            framet, flptst = video3d('hollywood/'+self.vid_path + '/' + test_name, (110,110,self.timesteps),
                                    color=self.color,skip=self.skip,rand=self.random_sample,
                                    optical_flow=self.optical_flow,proj = self.proj)
            # (x_new,y_new),(x_old,y_old)  N x 2, N x 2
            # new = flpts[0][0]

            flptst = np.array(flptst)

            flptst_array.append(flptst.flatten())
        flptst_array = np.array(flptst_array)
        import pdb; pdb.set_trace()
        nbrs = NearestNeighbors(n_neighbors=11, algorithm='ball_tree').fit(flpts_array)
        distances, indices = nbrs.kneighbors(flptst_array)
        idxs = [np.median(ii) for ii in indices]
        pred = [a[0] for a in np.array(train_labels)[indices]]
        pred = mode(np.array(train_labels)[indices],axis = 1)[0]
        pred = np.squeeze(pred)
        # correct = (pred==np.array(test_labels)).sum()/len(pred)

        svm_mod = SVC(C=10000, kernel='rbf', degree=3)
        svm_mod.fit(flpts_array, np.array(train_labels))
        svm_mod.predict(flptst_array)
        import pdb; pdb.set_trace()

        return distance, indices


    def run_epochs(self):
        e_train_loss = []
        e_train_acc = []
        e_test_loss = []
        e_test_acc = []
        for i in range(self.epochs):
            train_loss, train_acc = self.train_test(self.train_path, self.test_path)

base_path = 'hollywood/annotations/'
vid_path = 'videoclips'
num_epochs = 50
color = False
random_sample = True
skip = False
timesteps_vid = 10

tr = Trainer(base_path+'train_clean.txt',base_path+'test_clean.txt',
                vid_path,timesteps_vid,num_epochs,color,
                random_sample,skip,optical_flow = True)
trl,tstl,tra,tsta = tr.run_epochs()
