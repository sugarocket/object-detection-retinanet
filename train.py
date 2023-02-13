
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.retinanet import retinanet
from nets.retinanet_training import FocalLoss
from utils.callbacks import LossHistory
from utils.dataloader import RetinanetDataset, retinanet_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch

'''
Notes:


1、The trained weight file is saved in the logs folder. It will be saved once for each epoch.
  After training according to the default parameters, there will be 100 weights.

2、The size of the loss value is used to judge whether it has converged. The more important thing is that there is a trend of convergence, that is, the verification set loss continues to decrease. If the verification set loss basically does not change, the model basically converges.
   The specific size of the loss value is meaningless. The large and small are only in the calculation method of the loss, not close to 0.

'''  
if __name__ == "__main__":

 
    #   Use GPU
   
    Cuda            = True
  
    classes_path    = 'model_data/voc_classes.txt'

    model_path      = 'logs/ep096-loss0.105-val_loss0.180.pth'
    ##"model_data/retinanet_resnet50.pth"
   
    #  shape size of input
 
    input_shape     = [600, 600]
    
   
    #  2 is resnet50
 
    phi             = 2
    
    #   pretrained = False，Freeze_Train = Fasle，
 
    pretrained      = False


     # Training is divided into two stages, namely the freezing stage and the unfreeze stage.
     # Freeze phase training parameters
     # At this time, the backbone of the model is frozen, and the feature extraction network does not change
     #Occupies a small amount of memory, only fine-tuning the network
     
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 8
    Freeze_lr           = 1e-4
    
    # Training parameters in the thawing phase
    #Occupies a large amount of video memory, and all network parameters will change
 
    UnFreeze_Epoch      = 100
    Unfreeze_batch_size = 4
    Unfreeze_lr         = 1e-5
 
    #   Whether to freeze training, the default is to freeze the main training first and then unfreeze the training.

    Freeze_Train        = True
    
    #   Used to set whether to use multiple threads to read data
     # After opening, it will speed up data reading
    num_workers         = 4

    #   Get image path and label
   
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    #   get classes & anchor

    class_names, num_classes = get_classes(classes_path)


    #   Create a retinanet model

    model = retinanet(num_classes, phi, pretrained)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    focal_loss      = FocalLoss()
    loss_history    = LossHistory("logs/")
    
    
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    # The main feature extraction network feature is general, freezing training can speed up the training speed
     # It can also prevent the weight from being destroyed at the beginning of training.
     # Init_Epoch is the initial generation
     # Freeze_Epoch is the generation of freeze training
     # UnFreeze_Epoch total training generation
    if True:
        batch_size  = Freeze_batch_size
        lr          = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch   = Freeze_Epoch
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The data set is too small for training. Please expand the data set.")
        
        optimizer       = optim.Adam(model_train.parameters(), lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset   = RetinanetDataset(train_lines, input_shape, num_classes, train = True)
        val_dataset     = RetinanetDataset(val_lines, input_shape, num_classes, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=retinanet_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=retinanet_dataset_collate)

      
        #   Freeze a certain part of training
        
        if Freeze_Train:
            for param in model.backbone_net.parameters():
                param.requires_grad = False
                
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, focal_loss, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
            
    if True:
        batch_size  = Unfreeze_batch_size
        lr          = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch   = UnFreeze_Epoch
                        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The data set is too small for training. Please expand the data set.")
        
        optimizer       = optim.Adam(model_train.parameters(), lr)
        lr_scheduler    = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset   = RetinanetDataset(train_lines, input_shape, num_classes, train = True)
        val_dataset     = RetinanetDataset(val_lines, input_shape, num_classes, train = False)
        gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=retinanet_dataset_collate)
        gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=retinanet_dataset_collate)

        #   Freeze a certain part of training
        if Freeze_Train:
            for param in model.backbone_net.parameters():
                param.requires_grad = True
                
        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch(model_train, model, focal_loss, loss_history, optimizer, epoch, 
                    epoch_step, epoch_step_val, gen, gen_val, end_epoch, Cuda)
            lr_scheduler.step()
