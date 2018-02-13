
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import image
import os.path
import numpy as np 

 

def transform(data, target_wd, target_ht):
    data = mx.image.imresize(data, target_wd, target_ht)
    data = nd.transpose(data, (2,0,1))
    data = data.astype(np.float32)/127.5 - 1
    
    # if image is greyscale, repeat 3 times to get RGB image.
    if data.shape[0] == 1:
        data = nd.tile(data, (3, 1, 1))

    return data.reshape((1,) + data.shape)

def dataloader(path,width,heith):
    data_list = []
    for path,_, fnames in os.walk(path):
        for fname in fnames:
          #  if not fname.endswith('.bmp') or fname.endswith('.jpg'):
          #      continue
            img = os.path.join(path, fname)
            img_arr = mx.image.imread(img)
            img_arr = transform(img_arr, width, heith)
 
            data_list.append((img_arr))
    return data_list


def LoadDataSet(data_path,label_path,width,heith,batch_size):
    data = dataloader(data_path,width,heith)
    label = dataloader(label_path,width,heith)
    print('total num_data: ' + str(len(data)))
    print('total num_label: ' + str(len(label))) 
    DataSet = mx.io.NDArrayIter(data=nd.concatenate(data),label=nd.concatenate(label), batch_size=batch_size)
    
    return DataSet


 

