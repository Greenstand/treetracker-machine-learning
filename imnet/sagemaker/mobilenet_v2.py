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
# import s3fs




logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py


tree_synsets = {
    "judas": "n12513613",
    "palm": "n12582231",
    "pine": "n11608250",
    "china tree": "n12741792",
    "fig": "n12401684",
    "cabbage": "n12478768",
    "cacao": "n12201580",
    "kapok": "n12190410",
    "iron": "n12317296",
    "linden": "n12202936",
    "pepper": "n12765115",
    "rain": "n11759853",
    "dita": "n11770256",
    "alder": "n12284262",
    "silk": "n11759404",
    "coral": "n12527738",
    "huisache": "n11757851",
    "fringe": "n12302071",
    "dogwood": "n12946849",
    "cork": "n12713866",
    "ginkgo": "n11664418",
    "golden shower": "n12492106",
    "balata": "n12774299",
    "baobab": "n12189987",
    "sorrel": "n12242409",
    "Japanese pagoda": "n12570394",
    "Kentucky coffee": "n12496427",
    "Logwood": "n12496949"
}
nontree_synsets = {
    "garbage_bin": "n02747177",
    "carion_fungus": "n13040303",
    "basidiomycetous_fungus": "n13049953",
    "jelly_fungus": "n13060190",
    "desktop_computer": "n03180011",
    "laptop_computer": "n03642806",
    "cellphone": "n02992529",
    "desk": "n03179701",
    "station_wagon": "n02814533",
    "pickup_truck": "n03930630",
    "trailer_truck": "n04467665"
}
synsets = {**tree_synsets, **nontree_synsets}



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





class ImnetDataset(Dataset):

    # initialise function of class
    def __init__(self, dir, synsets, transforms=None, device=None, one_hot=False, nontrees=False):
        # the data directories
        self.img_dir = os.path.join(dir, "original_images")
        self.bb_dir = os.path.join(dir, "bounding_boxes")
        self.nontrees_present = nontrees
        # synsets library to get the associated class
        if not self.nontrees_present:  # only tree images
            self.synsets = tree_synsets
        else:  # mix other things
            self.synsets = synsets
        self.rev_synsets = {y: x for x, y in zip(synsets.keys(), synsets.values())}
        self.classes = list(self.synsets.keys())

        self.one_hot = one_hot
        self.imgs = []
        self.file_stream = io.StringIO()

        for i in self.classes:
            temp_imgs = os.listdir(os.path.join(self.img_dir, i))
            for img_path in temp_imgs:
                if not "tar" in img_path:
                    name = os.path.basename(img_path.split('.')[0])
                    self.imgs.append(name)

        self.bb_dict = {}
        for f, _, d in os.walk(self.bb_dir):
            for file in d:
                if os.path.splitext(file)[1] == ".xml" and file.split("_")[0] in tree_synsets.values():
                    with open(os.path.join(f, file)) as file_obj:
                        tree = ElementTree.parse(file_obj)
                        root = tree.getroot()
                        obj = root.find("object")
                        b = obj.find("bndbox")
                        xmin = int(b.find("xmin").text)
                        ymin = int(b.find("ymin").text)
                        xmax = int(b.find("xmax").text)
                        ymax = int(b.find("ymax").text)
                        self.bb_dict[os.path.join(f, file)] = (xmin, ymin, xmax, ymax)

        self.transforms = transforms
        self.device = device

    def __getitem__(self, idx):
        name = self.imgs[idx]
        label = self.rev_synsets[name.split("_")[0]]
        # modify filters to determine if trees present
        is_tree = 1.0
        if self.nontrees_present:
            if label in tree_synsets.keys():
                is_tree = 1.0
            else:
                is_tree = 0.0

        img_path = os.path.join(self.img_dir, label, f"{name}.JPEG")
        bb_path = os.path.join(self.bb_dir, label, "Annotation", name.split("_")[0], f"{name}.xml")
        with open(img_path) as f:
            img = Image.open(f).convert("RGB")

        if bb_path in self.bb_dict.keys():
            xmin, ymin, xmax, ymax = self.bb_dict[bb_path]
        else:
            # the whole image is the bounding box label, as NoneType was causing collating issue.
            xmin = 0
            ymin = 0
            xmax = img.size[0]
            ymax = img.size[1]
        boxes = torch.as_tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
        if not is_tree:
            boxes = torch.as_tensor([0, 0, 0, 0],
                                    dtype=torch.float32)  # 0 out nontree bounding boxes, don't want predictions for these

        if self.transforms is not None:
            img = self.transforms(img)

        if self.one_hot:
            image_id = torch.zeros(len(self.classes), dtype=torch.float32)
            image_id[self.classes.index(label)] = 1.0
        else:
            image_id = torch.tensor([self.classes.index(label)])

        targets = {}
        targets["boxes"] = boxes
        targets["image_class"] = image_id
        targets["is_tree"] = is_tree

        return img, targets

    def __len__(self):
        return len(self.imgs)

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
            nn.Linear(1280, 1),  # 1280 is num_outputs of the last feature layer
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


def loss_spec(binary_label, binary_preds, bbox_coords, bbox_preds, binary_weight=0.5, bbox_weight=0.5):
    '''
    The loss function we use is a weighted combination of binary loss and bounding box regression loss.
    Modify this function for other loss functions depending on use case.
    :param binary_label: Label whether image contains positive class
    :param binary_preds: Predict positive class presence in image
    :param bbox_coords: Label of bounding box in positive class. Only applies if positive class is present
    :param bbox_preds: Prediction of bounding box of positive class
    :return:
    '''

    binary_detection_error = nn.BCEWithLogitsLoss(binary_label, binary_preds.unsqueeze(1))# output, target
    bounding_box_error = nn.MSELoss(bbox_preds, bbox_coords)
    return binary_weight * binary_detection_error + bbox_weight * bounding_box_error


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(Customized_MobileNet(pretrained_model=models.mobilenet_v2(pretrained=True)))
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model.to(device)


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)


class ModelTrainer():
    '''
    An abstraction to help keep track of model parameters and run training.
    '''

    def __init__(self, model):

        self.model = model  # like Customized_MobileNet
        # Initialize device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.start_epoch = 0



    def train(self, *args):
        '''
        Main function to train.
        '''

        if self.device == torch.device("cuda:0"):
            self.model.cuda()
            # Define data loader for training and validation
        self.batch_size = args.batch_size
        self.model_savepath = args.model_dir
        dataset = ImnetDataset(args.training_dir, device=self.device)

        # Make validation split
        self.trainsize = int(args.train_split * len(dataset))
        self.valsize = len(dataset) - self.trainsize
        train_dataset, valid_dataset = torch.utils.data.dataset.random_split(dataset, [self.trainsize, self.valsize])
        self.data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, sampler=None,
                                      batch_sampler=None, num_workers=args.n_workers, collate_fn=None,
                                      pin_memory=args.pin_memory, drop_last=False, timeout=0,
                                      worker_init_fn=None)

        self.val_data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, sampler=None,
                                          batch_sampler=None, num_workers=args.n_workers, collate_fn=None,
                                          pin_memory=args.pin_memory, drop_last=False, timeout=0,
                                          worker_init_fn=None)


        # Loss specifications and optimizer parameter setting
        # This is defined here so that the underlying model can be changed (i.e. hidden layers)
        cps = [param for param in self.model.classifier.parameters()]
        rps = [param for param in self.model.regressor.parameters()]
        self.optimizer = torch.optim.Adam(params=cps + rps, lr=args.lr, weight_decay=args.gamma)

        if os.path.exists(self.model_savepath):
            print("Found saved model at savepath %s" % (self.model_savepath))
            checkpoint = torch.load(self.model_savepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
        else:
            pass

        is_distributed = len(args.hosts) > 1 and args.backend is not None
        logger.debug("Distributed training - {}".format(is_distributed))
        use_cuda = args.num_gpus > 0
        logger.debug("Number of gpus available - {}".format(args.num_gpus))
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
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
        num_tr_batches = np.ceil(self.trainsize / self.data_loader.batch_size)
        num_val_batches = np.ceil(self.valsize / self.valsize)
        epoch_loss = []
        epoch_acc = []
        epoch_rmse = []
        epoch_iou = []
        val_epoch_loss = []
        val_epoch_acc = []
        val_epoch_rmse = []
        val_epoch_iou = []
        print("Starting at epoch %d" % self.start_epoch)
        for epoch in range(1, args.epochs):
            epoch_start = time.time()
            print("=" * 50)
            logger.info("EPOCH ", epoch)
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
                    batchy["boxes"] = batchy["boxes"].cuda(non_blocking=True)
                    batchy["image_class"] = batchy["image_class"].cuda(non_blocking=True)
                    batchy["is_tree"] = batchy["is_tree"].cuda(non_blocking=True)
                class_labels = batchy["image_class"]
                box_labels = batchy["boxes"]
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
                    logger.info("\nLast Batch Avg Metrics, Batch %d/%d" % (batch_count, num_tr_batches))
                    logger.info("Total Loss: {:.3f}".format(
                        torch.mean(torch.as_tensor(batch_loss, dtype=torch.float32))))
                    logger.info("Classification Acc: {:.3f}".format(
                        torch.mean(torch.as_tensor(batch_acc, dtype=torch.float32))))
                    logger.info("BBox RMSE: {:.3f}".format(
                        torch.mean(torch.as_tensor(batch_rmse, dtype=torch.float32))))
                    logger.info("Avg Bbox IoU: {:.3f} \n".format(
                        torch.mean(torch.as_tensor(batch_iou, dtype=torch.float32))))
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    },
                        self.model_savepath)
                    print("Checkpoint created")

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
                            batchy["boxes"] = batchy["boxes"].cuda(non_blocking=True)
                            batchy["image_class"] = batchy["image_class"].cuda(non_blocking=True)
                            batchy["is_tree"] = batchy["is_tree"].cuda(non_blocking=True)
                        class_labels = batchy["image_class"]
                        box_labels = batchy["boxes"]
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
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tr_metric_dict': tr_metric_dict,
            'val_metric_dict': val_metric_dict
        },
            self.model_savepath)
        print("Final checkpoint created. Model dict and metrics saved. ")
        return self.model, tr_metric_dict, val_metric_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.0001, metavar='L',
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--pin_memory', type=bool, default=False, metavar='N',
                        help='pin memory')
    parser.add_argument('--n_workers', type=int, default=1, metavar='N',
                        help='number of dataloader workers')
    parser.add_argument('--val_interval', type=int, default=1, metavar='N',
                        help='how often to update validation metrics')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training-dir', type=str, default=os.environ['SM_INPUT_DIR'])
    # parser.add_argument('--test-dir', type=str, default=os.environ['SM_CHANNEL_TEST'])

    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    mobilenet_preprocessing = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    trainer = ModelTrainer(Customized_MobileNet(pretrained_model=models.mobilenet_v2(pretrained=True)))
    trainer.train(parser.parse_args())

