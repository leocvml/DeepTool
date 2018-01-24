import numpy as np
from skimage import data, io 
import matplotlib.pyplot as plt
import os.path

import cv2

data_path = 'data1/'
save_path = 'save/'

def DataLoader(path,as_grey = False):
    file_name =[]
    file_name = [x for x in os.listdir(path)]
    imgName_list = []
    img_list = []
    for name in file_name:
        imgName_list.append(name)
    img_list = np.array(img_list)

   
    return imgName_list

DataNameList = DataLoader(data_path) 
print('Loading Data OK!!!!')


class LabelTool:
    def __init__(self,DataNameList,num_class,norm=False):
        self.DataNameList = DataNameList
        drawing = False # true if mouse is pressed
        self.mode = True # if True, draw rectangle. Press 'm' to toggle to curve
        ix,iy = -1,-1
        self.delta = 1
        self.num_class = num_class 
        self.color_space=[(255,0,0),(0,255,0),(0,0,255),(128,128,0),(0,128,128),(128,0,128),(255,255,0),(255,0,255),(0,255,255),(255,255,255)]
        self.info =''
        self.file = open('test.txt' , 'w+')
        self.norm = norm
    # mouse callback function
    def mouse_event(self,event,x,y,flags,param):
        global ix,iy,drawing,mode,delta,num_class,color_space

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y


        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if self.mode == True:
                
                cv2.rectangle(self.img,(ix,iy),(x,y),self.color_space[self.delta],4)
                if self.norm:
                    self.info +=(str(self.delta)+'\t'+str(ix/self.img.shape[1])+'\t'+str(iy/self.img.shape[0])+'\t'+str(x/self.img.shape[1])+'\t'+str(y/self.img.shape[0])+'\t')
                else:
                    self.info +=(str(self.delta)+'\t'+str(ix)+'\t'+str(iy)+'\t'+str(x)+'\t'+str(y)+'\t')
            else:
                global img
                cv2.circle(self.img,(x,y),5,self.color_space[self.delta],-1)


    def Draw_label(self):
        for name in self.DataNameList:
            
            self.img  = cv2.imread(data_path + name)    
            cv2.namedWindow('image')
            cv2.setMouseCallback('image',self.mouse_event)
            self.info +=(name + '\t')
            
            while(1):
                cv2.imshow('image',self.img)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('m'):
                    mode = not mode
            
                if k == ord('r'):
                    self.img  = cv2.imread(data_path + name)
                    self.info = ''
                    cv2.imshow('image',self.img)
            
                if k == ord('q'):
                    self.delta = self.delta + 1 
                    self.delta = self.delta % self.num_class
                    print(self.delta)

                if k == ord('w'):
                    self.delta = self.delta - 1
                    self.delta = self.delta % self.num_class
                    print(self.delta)

                if k == 9:
                    cv2.imwrite(save_path + name , self.img)
                    self.info +=('\n')
                    self.file.write(self.info)
                    print('save file at:' + save_path + name)
                    self.info = ''
                    break

                if k == 27:
                    self.file.close()
                    exit()
        self.file.close()
        
        cv2.destroyAllWindows()

        
Label = LabelTool(DataNameList[:2],num_class = 10,norm = True)  #num_class  current limit :10
Label.Draw_label()





 






 




