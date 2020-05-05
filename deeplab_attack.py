import torch
import torch.nn as nn
import numpy as np
import easydict
from torch.autograd import Variable
from torch.utils import data
from deeplab_model import Res_Deeplab
#from deeplab_functions import get_iou
#from deeplab_functions import show_all
#from deeplab_functions import show_attack
#from deeplab_functions import _pickle_method
#from deeplab_functions import ConfusionMatrix
#from deeplab_functions import custom_IOU
#from deeplab_functions import VOCDataSet
#from deeplab_functions import show_deeplab
import deeplab_functions as dlf


import warnings
warnings.simplefilter('ignore')

# CUDA -> change to device 0 or 1
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {'num_workers': 4}

# Evaluation setup
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

# DATA_DIRECTORY = 'data/VOCdevkit/VOC2012'
# DATA_LIST_PATH = './dataset/list/val.txt'

DATA_DIRECTORY = '/home/m3kowal/Desktop/Research/Datasets/VOC2012/data/VOCdevkit/VOC2012'
DATA_LIST_PATH = '/home/m3kowal/Desktop/Research/Datasets/VOC2012/dataset/list/val.txt'

IGNORE_LABEL = 255
NUM_CLASSES = 21
NUM_STEPS = 1449 # Number of images in the validation set is 1449
RESTORE_FROM = './deeplab_snapshots/VOC12_scenes_20000.pth'

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    #parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    '''parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    '''
    args = easydict.EasyDict({
    "data_dir": DATA_DIRECTORY,
    "data_list": DATA_LIST_PATH,
    "ignore_label": IGNORE_LABEL,
    "num_classes": NUM_CLASSES,
    "restore_from": RESTORE_FROM,
    "gpu": 1
    })
    
    return args

# defining dataloader and upsampling function
args = get_arguments()
deeplab_model = Res_Deeplab(num_classes = 21)
saved_state_dict = torch.load(args.restore_from)
deeplab_model.load_state_dict(saved_state_dict)
deeplab_model.eval()
deeplab_model.cuda(1)

VOCDataSet2 = dlf.VOCDataSet(args.data_dir,args.data_list, crop_size=(505, 505), mean=IMG_MEAN)
testloader = data.DataLoader(VOCDataSet2, batch_size=1, shuffle=False, pin_memory=True)
interp = nn.Upsample(size=(505, 505), mode='bilinear', align_corners=True)
data_list = []

# image selection process
image_index = 0
for index, batch in enumerate(testloader):
    if index == image_index:
        image, label, size, name = batch 
        size = size[0].numpy()
        pred = deeplab_model(Variable(image).cuda(1))
        pred = interp(pred).cpu().data[0].numpy()
        pred = pred[:,:size[0],:size[1]]
        gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
        pred = pred.transpose(1,2,0)
        pred = np.asarray(np.argmax(pred, axis=2), dtype=np.int)
        img = image
        #show_all(gt, pred)
        #dlf.tell_deeplab(img, gt, pred, size)
        break

_ = dlf.attack_DeepLab(deeplab_model, image_index ,img, label, interp, size, num_pix = 15, iters=75, pop_size=250)
_ = dlf.attack_DeepLab(deeplab_model, image_index ,img, label, interp, size, num_pix = 20, iters=75, pop_size=250)
_ = dlf.attack_DeepLab(deeplab_model, image_index ,img, label, interp, size, num_pix = 30, iters=75, pop_size=250)
_ = dlf.attack_DeepLab(deeplab_model, image_index ,img, label, interp, size, num_pix = 50, iters=75, pop_size=250)

