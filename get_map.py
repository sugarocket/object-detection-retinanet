import os
import xml.etree.ElementTree as ET

from PIL import Image
from tqdm import tqdm

from retinanet import Retinanet
from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map

if __name__ == "__main__":

    #The entire map calculation process, including obtaining prediction results, obtaining real frames, and calculating VOC_map
  
    classes_path    = 'model_data/voc_classes.txt'
   
    #MINOVERLAP is used to specify the mAP0.x you want to obtain
  
    MINOVERLAP      = 0.5
    

    #map_vis is used to specify whether to enable the visualization of VOC_map calculations
    map_vis         = False
  
    
    VOCdevkit_path  = 'VOCdevkit'
    
    #Result output in file map_out
    map_out_path    = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)


    #obtaining prediction results

    print("Load model.")
    retinanet = Retinanet(confidence = 0.01, nms_iou = 0.5)
    print("Load model done.")

    print("Get predict result.")
    for image_id in tqdm(image_ids):
        image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".jpg")
        image       = Image.open(image_path)
        if map_vis:
            image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".jpg"))
        retinanet.get_map_txt(image_id, image, class_names, map_out_path)
    print("Get predict result done.")
        
    #obtaining real frames
    print("Get ground truth result.")
    for image_id in tqdm(image_ids):
        with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
            root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
            for obj in root.findall('object'):
                difficult_flag = False
                if obj.find('difficult')!=None:
                    difficult = obj.find('difficult').text
                    if int(difficult)==1:
                        difficult_flag = True
                obj_name = obj.find('name').text
                if obj_name not in class_names:
                    continue
                bndbox  = obj.find('bndbox')
                left    = bndbox.find('xmin').text
                top     = bndbox.find('ymin').text
                right   = bndbox.find('xmax').text
                bottom  = bndbox.find('ymax').text

                if difficult_flag:
                    new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                else:
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
    print("Get ground truth result done.")

    #calculating VOC_map
    print("Get map.")
    get_map(MINOVERLAP, True, path = map_out_path)
    print("Get map done.")

    