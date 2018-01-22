import numpy as np
from skimage import data, io 
import matplotlib.pyplot as plt



class SegEvaluation:
    def __init__(self, PredictImg, GroundTruthImg,inverse = False, norm = False):
        self.p = PredictImg
        self.g = GroundTruthImg


        if self.p.shape != self.g.shape:
            raise ValueError('shape different')
        if norm : #scale != 1
            self.p = self.p - self.p.min()
            self.p = self.p / (self.p.max() - self.p.min())
            
            self.g = self.g - self.g.min()
            self.g = self.g / (self.g.max() - self.g.min())
        
        if inverse:  # white = 0  
            self.p = 1 - self.p
            self.g = 1 - self.g
       #######################
       ##
       ##  GT = FN + TP
       ##  Predict = TP + FP
       ## 
       ##  TP = GT & Predict
       ##  FP  = Predict - TP
       ##  FN = GT - TP
       ##
       #######################
        self.p = np.asarray(self.p).astype(np.bool)
        self.g = np.asarray(self.g).astype(np.bool)

    
    def dice_coefficient(self):
        #   2 * TP / (FN + (2 * TP) + FP)
        #
        intersection = np.logical_and(self.p, self.g)
        
        return 2. * intersection.sum() / (self.p.sum() + self.g.sum())
  
        
    def IoU(self):
        # jaccard  TP / TP + FN +FP
        #
        TP = np.logical_and(self.p, self.g)

        return TP.sum() / (self.g.sum() + self.p.sum() - TP.sum())

    def recall(self):
        # recall sensitivity
        #
        #   TP / GT
        #
        
        TP = np.logical_and(self.p, self.g)
        return TP.sum() / self.g.sum()

    def precision(self):
        #
        #   TP / Predict
        #
        TP = np.logical_and(self.p, self.g)
        return TP.sum() / self.p.sum()
        


    
aimg = io.imread('1.jpg', as_grey = True)
bimg = io.imread('2.jpg', as_grey = True)


evaluation = SegEvaluation(aimg,bimg,inverse = True, norm = True)

print(evaluation.dice_coefficient())
print(evaluation.IoU())
print(evaluation.recall())
print(evaluation.precision())






























