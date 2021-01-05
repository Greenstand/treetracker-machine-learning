'''
This is an exmaple of how to create a script Sagemaker can use to train a custom-defined model on.

Created by Shubhom, December 2020
shubhom.bhattacharya@greenstand.org

'''


# Python libraries
import argparse
import json
import logging
import os
import sys

# PyTorch Libraries
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torchvision.models as models


# Utility Libraries
import time
from xml.etree import ElementTree
from PIL import Image, ImageDraw
from  collections import OrderedDict
import numpy as np
import io
import pandas as pd




# Helper functions
def rmse(x, y):
  '''
  Root-mean squared error of two vectors of the same batch
  '''
  return torch.sqrt( (1/x.size()[0]) * torch.sum((x-y) **2))

def iou(box_a, box_b):
  # order is xmin, ymin, xmax, ymax
  intersect_xmin = max(box_a[0], box_b[0])
  intersect_ymin = max(box_a[1], box_b[1])
  intersect_xmax = min(box_a[2], box_b[2])
  intersect_ymax = min(box_a[3], box_b[3])
  area_intersect = max(0, intersect_xmax - intersect_xmin) * max(0, intersect_ymax - intersect_ymin)

  area_a = (box_a[3] - box_a[1]) * (box_a[2] - box_a[0])
  area_b = (box_b[3] - box_b[1]) * (box_b[2] - box_b[0])
  union = area_a + area_b - area_intersect
  return area_intersect / union

class Sagemaker_Imnet_Dataset(Dataset):
    def __init__(self, path):
        super().__init__()
        OK_FORMATS = [".jpg", ".png"]
        self.basepath = path
        self.images = []
        self.labels_df = pd.read_csv(os.path.join(path, "labels.csv"), index_col=0)
        unlabeled_count = 0
        for f, _, d in os.walk(self.basepath):
            for file in d: 
                if os.path.splitext(file)[1] in OK_FORMATS:
                    if os.path.splitext(file)[0] in self.labels_df.index:
                        self.images.append(os.path.join(f, file))
                    else:
                        unlabeled_count += 1
        print ("Didn't find labels for %d images in %s"%(unlabeled_count, path))
        print ("Shuffled label preview")
        print (self.labels_df.sample(frac=1).head(5))
        print (self.labels_df.shape)
        
    def __getitem__(self, idx):
        imname = os.path.basename(self.images[idx])
        path_id = imname.split(".")[0]
        if "_aug" in path_id: # augmented
            path_id = imname.split("_aug")[0]
        row = self.labels_df.loc[path_id, :]
        class_label, bbox, is_tree = row["class"], row["bbox"], row["is_tree"]
        img = np.array(Image.open(self.images[idx]))
        binary = 0
        labels = {"species": class_label, "bbox": bbox, "is_tree": binary}
        return torch.as_tensor(img), labels
    
    def __len__(self):
        return len(self.images)
        
class Customized_MobileNet(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.pretrained = pretrained_model
        self.pretrained.classifier = nn.Identity()
        for param in self.pretrained.parameters():
            param.requires_grad = False
        self._binary_classifier_layer()
        self._regressor_layer()

    def forward(self, x):
        """
        The model is performing a regression on bounding boxes and a classifier
        """
        return self.classifier(self.pretrained(x)), self.regressor(self.pretrained(x))

    def _binary_classifier_layer(self):
        """
        Initializes final classification layer for labeling genus, species, etc.
        """
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 2),  # 1280 is num_outputs of the last feature layer, 2 indices for positive/negative class
        )
        for param in self.classifier.parameters():
            param.requires_grad = True

    def _regressor_layer(self):
        """
        A bounding box output layer for predicting object location
        This is currently designed to output exactly one bounding box
        """
        self.regressor = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, 4)  # 1280 is num_outputs of the last feature layer
        )
        for param in self.regressor.parameters():
            param.requires_grad = True

def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def loss_spec(binary_label, binary_preds, bbox_coords, bbox_preds, binary_weight=0.5):
    '''
    The loss function we use is a weighted combination of binary loss and bounding box regression loss.
    Modify this function for other loss functions depending on use case.
    :param binary_label: Label whether image contains positive class
    :param binary_preds: Predict positive class presence in image
    :param bbox_coords: Label of bounding box in positive class. Only applies if positive class is present
    :param bbox_preds: Prediction of bounding box of positive classf
    :return:
    '''
    # TODO: Add positive weight to remedy class imbalance
    binary_detection_error = nn.BCEWithLogitsLoss()
    bounding_box_error = nn.MSELoss()
    return binary_weight * binary_detection_error(binary_preds, binary_label) + (1-binary_weight) * bounding_box_error(bbox_preds, bbox_coords)


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Customized_MobileNet(pretrained_model=models.mobilenet_v2(pretrained=True)))
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, metrics, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)
    
    for k,v in metrics.items():
        json.dumps(v, os.path.join(model_dir, k))


class ModelTrainer():
    '''
    An abstraction to help keep track of model parameters and run training.
    '''

    def __init__(self, model):

        self.model = model  # like Customized_MobileNet
        # Initialize device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 0



    def train(self, args):
        '''
        Main function to train.
        '''

        if self.device == torch.device("cuda:0"):
            self.model.cuda()
            # Define data loader for training and validation
        self.batch_size = args.batch_size
        self.model_savepath = args.model_dir
        dataset = Sagemaker_Imnet_Dataset(os.environ["SM_CHANNEL_TRAINING"])
        val_dataset = Sagemaker_Imnet_Dataset(os.environ["SM_CHANNEL_VALIDATION"])
        
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, sampler=None,
                                      batch_sampler=None, num_workers=args.n_workers, collate_fn=None,
                                      pin_memory=args.pin_memory, drop_last=False, timeout=0,
                                      worker_init_fn=None)

        self.val_data_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True, sampler=None,
                                          batch_sampler=None, num_workers=args.n_workers, collate_fn=None,
                                          pin_memory=args.pin_memory, drop_last=False, timeout=0,
                                          worker_init_fn=None)


        # Loss specifications and optimizer parameter setting
        # This is defined here so that the underlying model can be changed (i.e. hidden layers)
        cps = [param for param in self.model.classifier.parameters()]
        rps = [param for param in self.model.regressor.parameters()]
        self.optimizer = torch.optim.Adam(params=cps + rps, lr=args.lr, weight_decay=args.gamma)
#       TODO: see how sagemaker handles model checkpointing; may be handled implicitly by SM Pytorch class
#         if os.path.exists(self.model_savepath):
#             print("Found saved model at savepath %s" % (self.model_savepath))
#             checkpoint = torch.load(self.model_savepath)
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#             self.start_epoch = checkpoint['epoch']
#         else:
#             pass

        is_distributed = (len(args.hosts) > 1) and (args.backend is not None)
        logger.debug("Distributed training - {}".format(is_distributed))
        use_cuda = args.num_gpus > 0
        logger.debug("Number of gpus available - {}".format(args.num_gpus))
        kwargs = {'num_workers': args.n_workers, 'pin_memory': True} if use_cuda else {}
        device = torch.device("cuda" if use_cuda else "cpu")

        if is_distributed:
            # Initialize the distributed environment.
            world_size = len(args.hosts)
            os.environ['WORLD_SIZE'] = str(world_size)
            host_rank = args.hosts.index(args.current_host)
            os.environ['RANK'] = str(host_rank)
            dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
            logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
                args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
                dist.get_rank(), args.num_gpus))

        # set the seed for generating random numbers
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)
        num_tr_batches = np.ceil(len(dataset) / self.data_loader.batch_size)
        num_val_batches = np.ceil(len(val_dataset) / self.val_data_loader.batch_size)
        epoch_loss = []
        epoch_acc = []
        epoch_rmse = []
        epoch_iou = []
        val_epoch_loss = []
        val_epoch_acc = []
        val_epoch_rmse = []
        val_epoch_iou = []
        logger.info("Starting at epoch {}".format(self.start_epoch))
        for epoch in range(1, args.epochs):
            epoch_start = time.time()
            print("=" * 50)
            logger.info("EPOCH {}".format(epoch))
            batch_count = 0
            batch_loss = []
            batch_acc = []
            batch_rmse = []
            batch_iou = []
            for batchx, batchy in self.data_loader:
                batch_count += 1
                # Device designation
                if self.device == torch.device("cuda:0"):
                    batchx = batchx.cuda(non_blocking=True)
                    batchy["bbox"] = batchy["bbox"].cuda(non_blocking=True)
                    batchy["species"] = batchy["species"].cuda(non_blocking=True)
                    batchy["is_tree"] = batchy["is_tree"].cuda(non_blocking=True)
                class_labels = batchy["species"]
                box_labels = batchy["bbox"]
                is_tree_labels = batchy["is_tree"]

                # Forward pass
                is_tree_preds, box_preds = self.model.forward(batchx)
                loss = loss_spec(is_tree_labels, is_tree_preds, box_labels, box_preds)
                loss.backward()
                self.optimizer.step()

                # Metrics
                box_rmse = rmse(box_preds, box_labels)
                avg_box_iou = torch.mean(
                    torch.as_tensor([iou(box_labels[i, :], box_preds[i, :]) for i in range(box_labels.size()[0])],
                                    dtype=torch.float32))
                binary_correct = (torch.round(is_tree_preds) == is_tree_labels.squeeze()).sum()
                acc = binary_correct / float(batchx.shape[0])
                batch_iou.append(avg_box_iou)
                batch_rmse.append(box_rmse)
                batch_acc.append(acc)
                batch_loss.append(loss.data)

                if batch_count % args.log_interval == 0 or batch_count == num_tr_batches:
                    logger.info("Last Batch Avg Metrics, Batch {}/{}".format(batch_count, num_tr_batches))
                    logger.info("Total Loss: {:.3f}".format(
                        torch.mean(torch.as_tensor(batch_loss, dtype=torch.float32))))
                    logger.info("Classification Acc: {:.3f}".format(
                        torch.mean(torch.as_tensor(batch_acc, dtype=torch.float32))))
                    logger.info("BBox RMSE: {:.3f}".format(
                        torch.mean(torch.as_tensor(batch_rmse, dtype=torch.float32))))
                    logger.info("Avg Bbox IoU: {:.3f}".format(
                        torch.mean(torch.as_tensor(batch_iou, dtype=torch.float32))))
#                     torch.save({
#                         'epoch': epoch + 1,
#                         'model_state_dict': self.model.state_dict(),
#                         'optimizer_state_dict': self.optimizer.state_dict(),
#                     },
#                         self.model_savepath)
#                     print("Checkpoint created")

            if epoch % args.val_interval == 0:
                print("VALIDATION EPOCH ", epoch)
                batch_count = 0

                self.model.eval()
                with torch.no_grad():
                    rmses = []
                    ious = []
                    losses = []
                    class_accs = []

                    for batchx, batchy in self.val_data_loader:
                        batch_count += 1
                        # Device designation
                        if self.device == torch.device("cuda:0"):
                            batchx = batchx.cuda(non_blocking=True)
                            batchy["bbox"] = batchy["bbox"].cuda(non_blocking=True)
                            batchy["species"] = batchy["species"].cuda(non_blocking=True)
                            batchy["is_tree"] = batchy["is_tree"].cuda(non_blocking=True)
                        class_labels = batchy["species"]
                        box_labels = batchy["bbox"]
                        is_tree_labels = batchy["is_tree"]
                        is_tree_preds, box_preds = self.model.forward(batchx)
                        losses.append(
                            loss_spec(is_tree_labels, is_tree_preds, box_labels, box_preds).data)
                        class_accs.append(float((torch.round(
                            is_tree_preds) == is_tree_labels.squeeze()).sum()) / self.val_data_loader.batch_size)
                        ious.append(torch.mean(torch.as_tensor(
                            [iou(box_labels[i, :], box_preds[i, :]) for i in range(box_labels.size()[0])],
                            dtype=torch.float32)))
                        rmses.append(rmse(box_preds, box_labels))

                    losses = torch.mean(torch.as_tensor(losses, dtype=torch.float32))
                    class_accs = torch.mean(torch.as_tensor(class_accs, dtype=torch.float32))
                    box_rmse = torch.mean(torch.as_tensor(rmses, dtype=torch.float32))
                    avg_box_iou = torch.mean(torch.as_tensor(ious, dtype=torch.float32))
                    val_epoch_loss.append(losses)
                    val_epoch_acc.append(class_accs)
                    val_epoch_rmse.append(box_rmse)
                    val_epoch_iou.append(avg_box_iou)

                    # We can change this to be epoch wise or not averaged over all batches
                    logger.info("Batch Average Val Loss: {:.3f}".format(losses))
                    logger.info("Batch Avg Val Classification Acc: {:.3f}".format(class_accs))
                    logger.info("Batch Avg Val BBox RMSE: {:.3f}".format(box_rmse))
                    logger.info("Batch Avg Avg Bbox IoU: {:.3f} \n".format(avg_box_iou))
                self.model.train()
            epoch_loss.append(torch.mean(torch.as_tensor(batch_loss, dtype=torch.float32)))
            epoch_acc.append(torch.mean(torch.as_tensor(batch_acc, dtype=torch.float32)))
            epoch_iou.append(torch.mean(torch.as_tensor(batch_iou, dtype=torch.float32)))
            epoch_rmse.append(torch.mean(torch.as_tensor(batch_rmse, dtype=torch.float32)))

            print("Epoch ", epoch + 1, " finished in ", time.time() - epoch_start)
        tr_metric_dict = {"Loss": epoch_loss, "Acc": epoch_acc, "IoU": epoch_iou, "RMSE": epoch_rmse}
        val_metric_dict = {"Loss": val_epoch_loss, "Acc": val_epoch_acc, "IoU": val_epoch_iou, "RMSE": val_epoch_rmse}
        metrics = {"train_metrics": tr_metric_dict, "val_metrics": val_metric_dict}
        save_model(self.model, metrics, os.environ["MODEL_DIR"])
        print("Final checkpoint created. Model dict and metrics saved. ")
        return self.model, tr_metric_dict, val_metric_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--train_split', type=float, default=0.75, metavar='N',
                       help='percentage of data to use as training vs validation')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.0001, metavar='L',
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--pin_memory', type=bool, default=False, metavar='N',
                        help='pin memory')
    parser.add_argument('--n_workers', type=int, default=0, metavar='N',
                        help='number of dataloader workers')
    parser.add_argument('--val_interval', type=int, default=1, metavar='N',
                        help='how often to update validation metrics')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    MOBILENET_PREPROCESSING = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    args = parser.parse_args()
    trainer = ModelTrainer(Customized_MobileNet(pretrained_model=models.mobilenet_v2(pretrained=True)))
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    trainer.train(args)

