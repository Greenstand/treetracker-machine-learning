import argparse
import json
import logging
import os
import sagemaker_containers
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

# Torch Dataset and IMNet Loading
import torch
from xml.etree import ElementTree
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from PIL import Image, ImageDraw
from  collections import OrderedDict
import numpy as np
import io


# Model development and training
import torchvision.models as models
import torch.nn as nn


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Based on https://github.com/pytorch/examples/blob/master/mnist/main.py

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


def _get_train_data_loader(dataset, batch_size, training_dir, is_distributed, **kwargs):
    logger.info("Get train data loader")
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset) if is_distributed else None
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train_sampler is None,
                                       sampler=train_sampler, **kwargs)


def _get_test_data_loader(dataset,test_batch_size, training_dir, **kwargs):
    logger.info("Get test data loader")
    return torch.utils.data.DataLoader(
        datasets.MNIST(training_dir, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)


def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def train(args):
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

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, is_distributed, **kwargs)
    test_loader = _get_test_data_loader(args.test_batch_size, args.data_dir, **kwargs)

    logger.debug("Processes {}/{} ({:.0f}%) of train data".format(
        len(train_loader.sampler), len(train_loader.dataset),
        100. * len(train_loader.sampler) / len(train_loader.dataset)
    ))

    logger.debug("Processes {}/{} ({:.0f}%) of test data".format(
        len(test_loader.sampler), len(test_loader.dataset),
        100. * len(test_loader.sampler) / len(test_loader.dataset)
    ))

    model = Customized_MobileNet(pretrained_model=models.mobilenet_v2(pretrained=True)).to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = _loss_specification(is_tree_labels, is_tree_preds, box_labels, box_preds)
            loss.backward()
            optimizer.step()
            loss =(output, target)
            loss.backward()
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.sampler),
                    100. * batch_idx / len(train_loader), loss.item()))
        test(model, test_loader, device)
    save_model(model, args.model_dir)


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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

    def __init__(self, model, learning_rate, device, batch_size, model_savepath,
                 gamma=1e-4, train_split=0.8, pin_memory=False, n_workers=0,
                 alpha=0.5, beta=0.5
                 ):

        self.model = model  # like Customized_MobileNet
        self.alpha = alpha
        self.beta = beta
        # Initialize device
        self.device = device
        if self.device == torch.device("cuda:0"):
            self.model.cuda()
        # Define data loader for training and validation
        self.batch_size = batch_size

        # Loss specifications and optimizer parameter setting
        # This is defined here so that the underlying model can be changed (i.e. hidden layers)
        cps = [param for param in self.model.classifier.parameters()]
        rps = [param for param in self.model.regressor.parameters()]
        self.optimizer = torch.optim.Adam(params=cps + rps, lr=learning_rate, weight_decay=gamma)
        self.binary_classification_criterion = nn.BCEWithLogitsLoss()
        self.regression_criterion = nn.MSELoss()

        if os.path.exists(self.model_savepath):
            print("Found saved model at savepath %s" % (self.model_savepath))
            checkpoint = torch.load(self.model_savepath)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch']
        else:
            self.start_epoch = 0

    def describe_training(self):
        print(self.trainsize, " training examples")
        print(self.valsize, " validation examples")
    #
    # def train(self, num_epochs, val_interval=1, batch_report=50, batch_lookback=10):
    #     '''
    #     Main function to train.
    #     @param num_epochs(int): Number of epochs of training
    #     @param val_interval(int): Interval epochs between validation metric
    #     @param batch_report(int): Interval batches between training reports
    #     @param batch_lookback(int): Number of batches to use for averaging metrics in printing
    #     '''
    #     num_tr_batches = np.ceil(self.trainsize / self.data_loader.batch_size)
    #     num_val_batches = np.ceil(self.valsize / self.valsize)
    #     epoch_loss = []
    #     epoch_acc = []
    #     epoch_rmse = []
    #     epoch_iou = []
    #     val_epoch_loss = []
    #     val_epoch_acc = []
    #     val_epoch_rmse = []
    #     val_epoch_iou = []
    #     print("Starting at epoch %d" % self.start_epoch)
    #     for epoch in range(self.start_epoch, num_epochs):
    #         epoch_start = time.time()
    #         print("=" * 50)
    #         print("EPOCH ", epoch)
    #         batch_count = 0
    #         batch_loss = []
    #         batch_acc = []
    #         batch_rmse = []
    #         batch_iou = []
    #         for batchx, batchy in self.data_loader:
    #             batch_count += 1
    #             # Device designation
    #             if self.device == torch.device("cuda:0"):
    #                 batchx = batchx.cuda(non_blocking=True)
    #                 batchy["boxes"] = batchy["boxes"].cuda(non_blocking=True)
    #                 batchy["image_class"] = batchy["image_class"].cuda(non_blocking=True)
    #                 batchy["is_tree"] = batchy["is_tree"].cuda(non_blocking=True)
    #             class_labels = batchy["image_class"]
    #             box_labels = batchy["boxes"]
    #             is_tree_labels = batchy["is_tree"]
    #
    #             # Forward pass
    #             is_tree_preds, box_preds = self.model.forward(batchx)
    #             loss = self._loss_specification(is_tree_labels, is_tree_preds, box_labels, box_preds)
    #             loss.backward()
    #             self.optimizer.step()
    #
    #             # Metrics
    #             box_rmse = rmse(box_preds, box_labels)
    #             avg_box_iou = torch.mean(
    #                 torch.as_tensor([iou(box_labels[i, :], box_preds[i, :]) for i in range(box_labels.size()[0])],
    #                                 dtype=torch.float32))
    #             binary_correct = (torch.round(is_tree_preds) == is_tree_labels.squeeze()).sum()
    #             acc = binary_correct / float(batchx.shape[0])
    #             batch_iou.append(avg_box_iou)
    #             batch_rmse.append(box_rmse)
    #             batch_acc.append(acc)
    #             batch_loss.append(loss.data)
    #
    #             if batch_count % batch_report == 0 or batch_count == num_tr_batches:
    #                 print("\nLast %d Batch Avg Metrics, Batch %d/%d" % (batch_lookback, batch_count, num_tr_batches))
    #                 print("Total Loss: {:.3f}".format(
    #                     torch.mean(torch.as_tensor(batch_loss[-batch_lookback:], dtype=torch.float32))))
    #                 print("Classification Acc: {:.3f}".format(
    #                     torch.mean(torch.as_tensor(batch_acc[-batch_lookback:], dtype=torch.float32))))
    #                 print("BBox RMSE: {:.3f}".format(
    #                     torch.mean(torch.as_tensor(batch_rmse[-batch_lookback:], dtype=torch.float32))))
    #                 print("Avg Bbox IoU: {:.3f} \n".format(
    #                     torch.mean(torch.as_tensor(batch_iou[-batch_lookback:], dtype=torch.float32))))
    #                 torch.save({
    #                     'epoch': epoch + 1,
    #                     'model_state_dict': self.model.state_dict(),
    #                     'optimizer_state_dict': self.optimizer.state_dict(),
    #                 },
    #                     self.model_savepath)
    #                 print("Checkpoint created")
    #
    #         if epoch % val_interval == 0:
    #             print("VALIDATION EPOCH ", epoch)
    #             batch_count = 0
    #
    #             self.model.eval()
    #             with torch.no_grad():
    #                 rmses = []
    #                 ious = []
    #                 losses = []
    #                 class_accs = []
    #
    #                 for batchx, batchy in self.val_data_loader:
    #                     batch_count += 1
    #                     # Device designation
    #                     if self.device == torch.device("cuda:0"):
    #                         batchx = batchx.cuda(non_blocking=True)
    #                         batchy["boxes"] = batchy["boxes"].cuda(non_blocking=True)
    #                         batchy["image_class"] = batchy["image_class"].cuda(non_blocking=True)
    #                         batchy["is_tree"] = batchy["is_tree"].cuda(non_blocking=True)
    #                     class_labels = batchy["image_class"]
    #                     box_labels = batchy["boxes"]
    #                     is_tree_labels = batchy["is_tree"]
    #                     is_tree_preds, box_preds = self.model.forward(batchx)
    #                     losses.append(
    #                         self._loss_specification(is_tree_labels, is_tree_preds, box_labels, box_preds).data)
    #                     class_accs.append(float((torch.round(
    #                         is_tree_preds) == is_tree_labels.squeeze()).sum()) / self.val_data_loader.batch_size)
    #                     ious.append(torch.mean(torch.as_tensor(
    #                         [iou(box_labels[i, :], box_preds[i, :]) for i in range(box_labels.size()[0])],
    #                         dtype=torch.float32)))
    #                     rmses.append(rmse(box_preds, box_labels))
    #
    #                 losses = torch.mean(torch.as_tensor(losses, dtype=torch.float32))
    #                 class_accs = torch.mean(torch.as_tensor(class_accs, dtype=torch.float32))
    #                 box_rmse = torch.mean(torch.as_tensor(rmses, dtype=torch.float32))
    #                 avg_box_iou = torch.mean(torch.as_tensor(ious, dtype=torch.float32))
    #                 val_epoch_loss.append(losses)
    #                 val_epoch_acc.append(class_accs)
    #                 val_epoch_rmse.append(box_rmse)
    #                 val_epoch_iou.append(avg_box_iou)
    #
    #                 # We can change this to be epoch wise or not averaged over all batches
    #                 print("Batch Average Val Loss: {:.3f}".format(losses))
    #                 print("Batch Avg Val Classification Acc: {:.3f}".format(class_accs))
    #                 print("Batch Avg Val BBox RMSE: {:.3f}".format(box_rmse))
    #                 print("Batch Avg Avg Bbox IoU: {:.3f} \n".format(avg_box_iou))
    #             self.model.train()
    #         epoch_loss.append(torch.mean(torch.as_tensor(batch_loss, dtype=torch.float32)))
    #         epoch_acc.append(torch.mean(torch.as_tensor(batch_acc, dtype=torch.float32)))
    #         epoch_iou.append(torch.mean(torch.as_tensor(batch_iou, dtype=torch.float32)))
    #         epoch_rmse.append(torch.mean(torch.as_tensor(batch_rmse, dtype=torch.float32)))
    #
    #         print("Epoch ", epoch + 1, " finished in ", time.time() - epoch_start)
    #     tr_metric_dict = {"Loss": epoch_loss, "Acc": epoch_acc, "IoU": epoch_iou, "RMSE": epoch_rmse}
    #     val_metric_dict = {"Loss": val_epoch_loss, "Acc": val_epoch_acc, "IoU": val_epoch_iou, "RMSE": val_epoch_rmse}
    #     torch.save({
    #         'epoch': epoch + 1,
    #         'model_state_dict': self.model.state_dict(),
    #         'optimizer_state_dict': self.optimizer.state_dict(),
    #         'tr_metric_dict': tr_metric_dict,
    #         'val_metric_dict': val_metric_dict
    #     },
    #         self.model_savepath)
    #     print("Final checkpoint created. Model dict and metrics saved. ")
    #     return self.model, tr_metric_dict, val_metric_dict

    def _loss_specification(self, is_tree_labels, is_tree_preds, box_labels, box_preds):
        binary_detection_error = self.binary_classification_criterion(is_tree_preds,
                                                                      is_tree_labels.unsqueeze(1))  # output, target
        bounding_box_error = self.regression_criterion(box_preds, box_labels)
        return self.alpha * binary_detection_error + self.beta * bounding_box_error
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())