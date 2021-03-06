# One pixel attack imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pathlib import Path
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms, models
from tqdm import tqdm

writer = SummaryWriter()
sns.set()
sns.set_style("dark")
sns.set_palette("muted")
sns.set_color_codes("muted")


# DeepLabv2 imports
import argparse
import scipy
import scipy.misc
from scipy import ndimage
import cv2
import easydict
import pickle
import copyreg
import types
from sklearn.metrics import jaccard_similarity_score

from torch.autograd import Variable
import torchvision
#import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.utils import data
from torchsummary import summary

import sys
import os
import os.path as osp
import random
import csv
import timeit
from multiprocessing import set_start_method, Pool
import collections
from collections import OrderedDict
affine_par = True


def get_iou(data_list, class_num, save_path=None):

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool() 
    m_list = pool.map(f, data_list)
    pool.close() 
    pool.join() 
    
    for m in m_list:
        ConfM.addM(m)
    
    aveJ, j_list, M = ConfM.jaccard()
    print('meanIOU: ' + str(aveJ) + '\n')
    
    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    for i in range(len(j_list)):
        print(classes[i], j_list[i])
    
    #if save_path:
     #   with open(save_path, 'w') as f:
            #f.write('meanIOU: ' + str(aveJ) + '\n')
            #f.write(str(j_list)+'\n')
            #f.write(str(M)+'\n')

def show_all(gt, pred):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2)
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5), 
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5), 
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0), 
                    (0.5,0.75,0),(0,0.25,0.5),(1,1,1)]

    cmap = colors.ListedColormap(colormap)
    bounds=[0,0.5,1.5,2.5,2.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,255]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('gt')
    ax1.imshow(gt, cmap=cmap, norm=norm)

    ax2.set_title('pred')
    ax2.imshow(pred, cmap=cmap, norm=norm)

    plt.show()
    
def show_attack(pred_og, pred_attack):
    import matplotlib.pyplot as plt
    from matplotlib import colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(1, 2, figsize=(15,15))
    ax1, ax2 = axes

    classes = np.array(('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor'))
    
    colormap = [(0,0,0),(0.5,0,0),(0,0.5,0),(0.5,0.5,0),(0,0,0.5),(0.5,0,0.5),(0,0.5,0.5), 
                    (0.5,0.5,0.5),(0.25,0,0),(0.75,0,0),(0.25,0.5,0),(0.75,0.5,0),(0.25,0,0.5), 
                    (0.75,0,0.5),(0.25,0.5,0.5),(0.75,0.5,0.5),(0,0.25,0),(0.5,0.25,0),(0,0.75,0), 
                    (0.5,0.75,0),(0,0.25,0.5),(1,1,1)]

    cmap = colors.ListedColormap(colormap)
    bounds=[0,0.5,1.5,2.5,2.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5,20.5,255]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    ax1.set_title('Prediction Before Attack')
    ax1.imshow(pred_og, cmap=cmap, norm=norm)
    
    ax2.set_title('Prediction After Attack')
    ax2.imshow(pred_attack, cmap=cmap, norm=norm)
    
    plt.show()
    return fig

# Metric setup
def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)
copyreg.pickle(types.MethodType, _pickle_method)

class ConfusionMatrix(object):
    def __init__(self, nclass, classes=None):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))

    def add(self, gt, pred):
        assert(np.max(pred) <= self.nclass)
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == 255:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert(matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall/self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy/self.nclass

    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))
        
        return np.sum(jaccard_perclass)/len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass: #and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m


def custom_IOU(pred, gt):
    pred = np.asarray(pred).astype(np.bool)
    gt = np.asarray(gt).astype(np.bool)
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    return intersection.sum() / float(union.sum())

# VOC Dataset 
class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=False, mirror=False, ignore_label=255):
        self.root = root #'/home/m3kowal/Desktop/Class/Project/Pytorch-Deeplab-master/VOCdevkit/VOC2012'
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters)/len(self.img_ids)))
        self.files = []
        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file   = osp.join(self.root, "JPEGImages/%s.jpg" % name)
            label_file = osp.join(self.root, "SegmentationClassAug/%s.png" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]
        
        #image2 = cv2.imread( datafiles["img"],   cv2.IMREAD_COLOR)
        image = cv2.imread( datafiles["img"],   cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        

        size = image.shape
        name = datafiles["name"]

        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w = label.shape
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))    
        
        else:
            img_pad, label_pad = image, label

        img_h, img_w = label_pad.shape
        
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        
        h_off = 0
        w_off = 0
        
        # roi = cv2.Rect(w_off, h_off, self.crop_w, self.crop_h);
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        #image = image[:, :, ::-1]  # change to BGR
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        return image.copy(), label.copy(), np.array(size), name
    

DEEPLAB_LABELS = ('background', 'aeroplane', 'bicycle', 'bird', 'boat','bottle', 'bus', 'car', 'cat', 'chair','cow', 
                  'diningtable', 'dog', 'horse','motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

def show_deeplab(img, size, save=False):
    npimg = img.cpu().data[0].numpy()
    npimg = npimg[:,:size[0],:size[1]]
    npimg = np.transpose(npimg[::-1,:,:], (1,2,0))
    npimg = (npimg-npimg.min())/(npimg.max()-npimg.min())

    fig = plt.figure(figsize=(15,15))
    plt.imshow(npimg)
    plt.show()
    return fig

def tell_deeplab(img, gt, pred, size, target_label=None):
    fig = show_deeplab(img, size)
    show_all(gt, pred)
    print('IOU = ', custom_IOU(gt, pred))


def perturb_deeplab(p, img):
    # Elements of p should be in range [0,1]
    #img = img[:,:,:size[0],:size[1]]
    img_size_h = img.size(2) # C x _H_ x W, assume H == W
    img_size_w = img.size(3)
    p_img = img.data[0].clone()

    peturb_location_xy = [((p[0] * img_size_w).astype(int)),((p[1]* img_size_h).astype(int)) ]
    peturb_location_xy[0] = np.clip(peturb_location_xy[0], 0, img_size_w-1)
    peturb_location_xy[1] = np.clip(peturb_location_xy[1], 0, img_size_h-1)
    rgb = p[2:5].copy()
    rgb = np.clip(rgb, 0, 1)
    p_img[:,peturb_location_xy[1],peturb_location_xy[0]] = torch.from_numpy(rgb)
    return p_img

def perturb_deeplab_multipix(p, img):
    # Elements of p should be in range [0,1]
    #img = img[:,:,:size[0],:size[1]]
    img_size_h = img.size(2) # C x _H_ x W, assume H == W
    img_size_w = img.size(3)
    p_img = img.data[0].clone()
    
    peturb_locations_xy = []
    for i in p:
        peturb_locations_xy.append( [((i[0] * img_size_w).astype(int)),((i[1]*img_size_h).astype(int))])
    
    for j in range(len(peturb_locations_xy)): 
        peturb_locations_xy[j] = np.clip(peturb_locations_xy[j], 0, img_size_w-1)
        peturb_locations_xy[j] = np.clip(peturb_locations_xy[j], 0, img_size_h-1)
    
    rgb = []
    for k in p:
        rgb.append(k[2:5].copy())
    rgb = np.clip(rgb, 0, 1)
    
    for n in range(len(p)):
        p_img[:,peturb_locations_xy[n][1],peturb_locations_xy[n][0]] = torch.from_numpy(rgb[n])

    return p_img

def deeplab_visualize_perturbation(p, img, model, size, target_label=None):
    p_img = perturb_deeplab_multipix(p, img)
    
    print("Perturbation:", p)
    fig = show_deeplab(p_img.unsqueeze(0), size)
    #tell_deeplab(p_img, label, model, target_label)
    return p_img, fig

def evaluate_DeepLab(candidates, img, label, model, interp, size):
    preds = []
    model.eval()
    with torch.no_grad():
        for i, xs in enumerate(candidates):
            p_img = perturb_deeplab_multipix(xs, img)
            pred = model(Variable(p_img.unsqueeze(0)).cuda(1))
            pred = interp(pred).cpu().data[0].numpy()
            pred = pred[:,:size[0],:size[1]]
            gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
            pred = pred.transpose(1,2,0)
            pred = np.asarray(np.argmax(pred, axis=2), dtype=np.int)
            #img = image
            #tell_deeplab(img, gt, pred)
            preds.append(custom_IOU(gt, pred))
    return np.array(preds)

# evolve function should work for both models
def evolve_DeepLab(candidates, F=0.5, strategy="clip"):
    gen2 = candidates.copy()
    num_candidates = len(candidates)
    for i in range(num_candidates):
        x1, x2, x3 = candidates[np.random.choice(num_candidates, 3, replace=False)]
        x_next = (x1 + F*(x2 - x3))
        if strategy == "clip":
            gen2[i] = np.clip(x_next, 0, 1)
        elif strategy == "resample":
            x_oob = np.logical_or((x_next < 0), (1 < x_next))
            x_next[x_oob] = np.random.random([5,5])[x_oob]
            gen2[i] = x_next
    return gen2

# just change the fitness function should do the trick
def attack_DeepLab(model, image_index, img, label, interp, size, num_pix=1, iters=100, pop_size=100, verbose=True):
    # Targeted: maximize target_label if given (early stop > 50%)
    # Untargeted: minimize true_label otherwise (early stop < 5%)
    experiment_directory = '/home/m3kowal/Desktop/Research/One Pixel Attack for Dense Prediction/experiments/img_num-pix_iters_pop-size' + str(image_index) + '-' + str(num_pix) + '-' + str(iters) + '-' + str(pop_size)

    if not os.path.exists(experiment_directory):
        os.makedirs(experiment_directory)
    
    candidates = np.random.random((pop_size,5, 5))
    
    for i in candidates:
        i[:,2:5] = np.clip(np.random.normal(0.5, 0.5, (5,3)), 0, 1)
    
    fitness = evaluate_DeepLab(candidates, img, label, model, interp, size)
    fitness_list = []
    def is_success():
        return  fitness.min() < 0.35
    
    for iteration in range(iters):
        # Early Stopping
        if is_success():
            break
        if verbose and iteration%5 == 0: # Print progress
            print("Target Probability [Iteration {}]:".format(iteration), fitness.min())
            fitness_list.append(fitness.min())
        # Generate new candidate solutions
        new_gen_candidates = evolve_DeepLab(candidates, strategy="resample")
        # Evaluate new solutions
        new_gen_fitness = evaluate_DeepLab(new_gen_candidates, img, label, model, interp, size)
        # Replace old solutions with new ones where they are better
        successors = new_gen_fitness < fitness
        candidates[successors] = new_gen_candidates[successors]
        fitness[successors] = new_gen_fitness[successors]
    best_idx = fitness.argmin()
    best_solution = candidates[best_idx]
    best_score = fitness[best_idx]
    if verbose:
        attack_img, peturbed_image = deeplab_visualize_perturbation(best_solution, img, model, size)
        peturbed_image.savefig(str(experiment_directory) + '/perturbed_image')

        pred_og = model(Variable(img).cuda(1))
        pred_og = interp(pred_og).cpu().data[0].numpy()
        pred_og = pred_og[:,:size[0],:size[1]]
        gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
        pred_og = pred_og.transpose(1,2,0)
        pred_og = np.asarray(np.argmax(pred_og, axis=2), dtype=np.int)
    
        p_img = perturb_deeplab_multipix(best_solution, img)


        pred = model(Variable(p_img.unsqueeze(0)).cuda(1))
        pred = interp(pred).cpu().data[0].numpy()
        pred = pred[:,:size[0],:size[1]]
        pred = pred.transpose(1,2,0)
        pred = np.asarray(np.argmax(pred, axis=2), dtype=np.int)
        
        seg_masks = show_attack(pred_og, pred)
        seg_masks.savefig(str(experiment_directory)+'/seg_masks')


        print('IOU BEFORE attack:', custom_IOU(gt, pred_og))
        print('IOU AFTER attack:', custom_IOU(gt, pred))

        IOU_Data = [[custom_IOU(gt, pred_og)], [custom_IOU(gt, pred)]]

        with open(str(experiment_directory)+'/IOU.csv', 'w') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerows(IOU_Data)

        csvFile.close()

    return is_success(), best_solution, best_score

