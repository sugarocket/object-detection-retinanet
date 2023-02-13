##Retinanet：
##Code running instructions  

 
 
##  
1. Performance 
2. Environment 
3. Download 
4. How to train 
5. How to predict 
6. How to eval 
 
 
 
## Performance 
| dataset | weight file | test dataset | size of input image |  mAP  | 

| VOC07+12 | retinanet_resnet50.pth | VOC-Test07 | 600x600 | 82.51 
 
 
## Environment 
torch==1.2.0
numpy

## Download 
 
VOC dataset http://host.robots.ox.ac.uk/pascal/VOC/voc2012/ 

 
## Train 
### train VOC07+12 dataset 
1. data preparation    
**Use the VOC format for training. Before training, need to download the VOC07+12 data set, decompress it and place it in the root directory called “VOCdevkit”, then named the dataset folder “VOC2007” inside the VOCdevkit**   
 
 
2. Data preprocessing    
 
 
The entire labeling process: run voc_annotation.py, it will generate 2007_train.txt and 2007_val.txt. 
 
![image](https://user-images.githubusercontent.com/78404450/218348242-de646d83-e9a3-4180-a702-8fba3125c502.png)


3. Start training 
•	Initially, set the model_path in train.py file as resnet50-19c8e357.pt which located in model_data
•	Run train.py start training, which will take many times cost me around 20 hours, after training multiple 100 epochs, the weights will be generated in the folder called logs,  
 ![image](https://user-images.githubusercontent.com/78404450/218348252-dbf9fd8f-7c5d-425e-85b8-bbd24772f81f.png)

 
4. Training result prediction 
Two files are needed to predict the training results, namely retinanet.py and predict.py.

 
## Prediction 
### Use weights after training 
1. Follow the training steps to train  
2. In the retinanet.py file, modify model_path and classes_path in the following sections to make them correspond to the trained files; **model_path corresponds to the weight file under the logs folder, and classes_path is the class corresponding to model_path**. 

I trained it by using cloud sever so I just simple download several of them and chose logs/ep096-loss0.105-val_loss0.180.pth to be my own new weight because it has one of the smallest loss within others.

3. Run predict.py for detection. After running, we can enter an image path to detect. e.g., images/test.jpg, and enter ‘end’ to kill the process.

  ![image](https://user-images.githubusercontent.com/78404450/218348264-993e0cea-e235-4f12-ad7e-ac1b25aa359c.png)

 
 
## Evaluation steps 
### a、Evaluate the test set of VOC07+12 
1. We use the VOC format for evaluation. VOC07+12 has divided the test set, no need to use voc_annotation.py to generate the txt in the ImageSets folder. 
2. After dividing the test set with voc_annotation.py, go to the get_map.py file to modify the classes_path. The classes_path is used to point to the txt corresponding to the detection category. 
3. Run get_map.py to get the evaluation result, which will be saved in the map_out folder. AP Recall and all other results are included in map_out/results folder.
 




