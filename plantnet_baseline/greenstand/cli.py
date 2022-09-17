import torch


def add_all_parsers(parser):
    _add_loss_parser(parser)
    _add_training_parser(parser)
    _add_model_parser(parser)
    _add_hardware_parser(parser)
    _add_misc_parser(parser)
    _add_dataset_parser(parser)
    _add_greenstand_parser(parser) # GREENSTAND


def _add_loss_parser(parser):
    group_loss = parser.add_argument_group('Loss parameters')
    group_loss.add_argument('--mu', type=float, default=0., help='weight decay parameter')


def _add_training_parser(parser):
    group_training = parser.add_argument_group('Training parameters')
    group_training.add_argument('--lr', type=float, help='learning rate to use')
    group_training.add_argument('--batch_size', type=int, default=256, help='default is 256')
    group_training.add_argument('--n_epochs', type=int)
    group_training.add_argument('--k', nargs='+', help='value of k for computing the topk loss and computing topk accuracy',
                                required=True, type=int)


def _add_model_parser(parser):
    group_model = parser.add_argument_group('Model parameters')
    group_model.add_argument('--model', choices=['resnet50', 'densenet121', 'densenet169', 'mobilenet_v2', 'inception_resnetv2'],
                             default='resnet50', help='choose the model you want to train on')


def _add_hardware_parser(parser):
    group_hardware = parser.add_argument_group('Hardware parameters')
    group_hardware.add_argument('--use_gpu', type=int, choices=[0, 1], default=torch.cuda.is_available())


def _add_dataset_parser(parser):
    group_dataset = parser.add_argument_group('Dataset parameters')
    group_dataset.add_argument('--size_image', type=int, default=256,
                               help='size you want to resize the images to')


def _add_misc_parser(parser):
    group_misc = parser.add_argument_group('Miscellaneous parameters')
    group_misc.add_argument('--seed', type=int, help='set the seed for reproductible experiments')
    group_misc.add_argument('--num_workers', type=int, default=4,
                            help='number of workers for the data loader. Default is one. You can bring it up. '
                                 'If you have memory errors go back to one')
    group_misc.add_argument('--root', help='location of the train val and test directories')
    group_misc.add_argument('--save_name_xp', help='name of the saving file')

def _add_greenstand_parser(parser): # GREENSTAND
    group_greenstand = parser.add_argument_group('Miscellaneous parameters')
    group_greenstand.add_argument('--visualize', type=str, help='set to True if you want to visualize data and model output')
    group_greenstand.add_argument('--bucket', type=str, help='this is the bucket where the greenstand images are in S3')
    group_greenstand.add_argument('--prefixes', type=str, help='csv. list all of the prefixes within that bucket that you want to sync')
    group_greenstand.add_argument('--sub_dir_limit', type=int, help='max images per dataset per species to sync locally')
    group_greenstand.add_argument('--local_path', type=str, help='this is where the greenstand images will reside on your local machine')
    group_greenstand.add_argument('--train_test_split', type=float, help='this is the first split done, between training and test')
    group_greenstand.add_argument('--train_val_split', type=float, help='this split is done on the already split training data to make train and val')
    group_greenstand.add_argument('--metadata_file', type=str, help='this will contain the information on the image split between train, val, and test')
    group_greenstand.add_argument('--preloaded_model_location', type=str, help='set blank if not using, else enter path to pretrained model')
