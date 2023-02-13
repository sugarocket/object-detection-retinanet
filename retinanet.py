import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.retinanet import retinanet
from utils.utils import (cvtColor, get_classes, preprocess_input,
                         resize_image)
from utils.utils_bbox import decodebox, non_max_suppression



class Retinanet(object):
    _defaults = {
       
        #  After training, there are multiple weight files in the logs folder, just choose the one with lower validation set loss.
        #The lower loss of the verification set does not mean that the mAP is higher, but only the better generalization 
        #performance of the weight on the verification set.
        
        ##modify model_path by using a fully trarined weight
        "model_path"        : 'logs/ep099-loss0.107-val_loss0.181.pth',
        "classes_path"      : 'model_data/voc_classes.txt',

        
        #   size of input images
        #---------------------------------------------------------------------#
        "input_shape"       : [600, 600],
        
        #   Used to select the version of the model used which is resnet50
     
        "phi"               : 2,
    
        #   Only prediction boxes with a score greater than the confidence level will be retained
   
        "confidence"        : 0.5,
       
        #   The size of nms_iou used for non-maximum suppression
       
        "nms_iou"           : 0.3,

        # This variable is used to control whether to use letterbox_image to resize the input image without distortion,

        "letterbox_image"   : True,
        #---------------------------------------------------------------------#
        #   
       
        "cuda"              : True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   Initialize Retinanet
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        #---------------------------------------------------#
        #  Calculate the total number of classes
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        
        #---------------------------------------------------#
        #   Set different colors for the picture frame
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()
        
    #---------------------------------------------------#
    #   Load model
    #---------------------------------------------------#
    def generate(self):
        #----------------------------------------#
        #   Create Retinanet model
        #----------------------------------------#
        self.net    = retinanet(self.num_classes, self.phi)

        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

  
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        #Here, the image is converted into RGB image to prevent grayscale image from reporting errors during prediction.
         # The code only supports the prediction of RGB images, all other types of images will be converted into RGB
        
        image       = cvtColor(image)
       
        #  Add gray bars to the image to achieve undistorted resize
         # You can also directly resize for identification
       
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)

        #Add the batch_size dimension, image preprocessing, and normalization.

        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
  
            _, regression, classification, anchors = self.net(images)
            
            #Decode the prediction result
            outputs     = decodebox(regression, anchors, self.input_shape)
            results     = non_max_suppression(torch.cat([outputs, classification], axis=-1), self.input_shape, 
                                    image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
               
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

     
        #   set font and thickness
     
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        
     
        #   Draw picture
    
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image


    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])

        #Here, the image is converted into RGB image to prevent grayscale image from reporting errors during prediction.
         # The code only supports the prediction of RGB images, all other types of images will be converted into RGB
        image       = cvtColor(image)

        #  Add gray bars to the image to achieve undistorted resize
        # You can also directly resize for identification
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        
        #Add the batch_size dimension, image preprocessing, and normalization.
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
     
            _, regression, classification, anchors = self.net(images)
            
            
            #  decode

            outputs     = decodebox(regression, anchors, self.input_shape)
            results     = non_max_suppression(torch.cat([outputs, classification], axis=-1), self.input_shape, 
                                    image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
               
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 5], dtype = 'int32')
            top_conf    = results[0][:, 4]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
