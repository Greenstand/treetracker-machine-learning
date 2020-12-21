'''
This script is intended to be an example of using Torchvision transforms for preprocessing in Sagemaker.

See how this works at:
https://docs.aws.amazon.com/sagemaker/latest/dg/processing-container-run-scripts.html

Created by Shubhom
December 2020
'''
# PyTorch Libraries
import torch
import torchvision as tv

import os
import matplotlib.pyplot as plt



resize = tv.transforms.Compose([
    tv.transforms.Resize(224),
    tv.transforms.CenterCrop(224),
    tv.transforms.ToTensor()])


sample = tv.datasets.ImageFolder(
    root=os.environ['SM_INPUT_DIR'],
    transform=tv.transforms.ToTensor())

sample = iter(sample)
sample_resized = iter(sample_resized)

fig, ax = plt.subplots(1, 2, figsize=(10,5))
image = next(iter(sample))[0]
image_resized = next(iter(sample_resized))[0]

ax[0].imshow(image.permute(1, 2, 0))
ax[0].axis('off')
ax[0].set_title(f'Before - {tuple(image.shape)}')
ax[1].imshow(image_resized.permute(1, 2, 0))
ax[1].axis('off')
ax[1].set_title(f'After - {tuple(image_resized.shape)}');
plt.tight_layout()

augment = tv.transforms.Compose([
    tv.transforms.RandomResizedCrop(224),
    tv.transforms.RandomHorizontalFlip(p=0.5),
    tv.transforms.RandomVerticalFlip(p=0.5),
    tv.transforms.ColorJitter(
        brightness=.2,
        contrast=.2,
        saturation=.2,
        hue=.2),
    tv.transforms.ToTensor()])




batch_size = 4
shuffle = True
num_workers = 4

dataloaders = {}
for s in splits:
    dataloaders[s] = torch.utils.data.DataLoader(
        datasets[s],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )



datasets = {}
for s in splits:
    datasets[s] = tv.datasets.ImageFolder(
        root = data_dir / s,
        transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.ToTensor()]
        )
    )

resized_path = pathlib.Path('./data_resized')
resized_path.mkdir(exist_ok=True)
for s in splits:
    split_path = resized_path / s
    split_path.mkdir(exist_ok=True)
    for idx, (img_tensor, label) in enumerate(tqdm(datasets[s])):
        label_path = split_path / f'{label:02}'
        label_path.mkdir(exist_ok=True)
        filename = datasets[s].imgs[idx][0].split('/')[-1]
        tv.utils.save_image(img_tensor, label_path / filename)

if pathlib.Path('pickled_data/pytorch_bucket_name.pickle').exists():
    with open('pickled_data/pytorch_bucket_name.pickle', 'rb') as f:
        bucket_name = pickle.load(f)
        print('Bucket Name:', bucket_name)
else:
    bucket_name = f'sagemaker-pytorch-ic-{str(uuid.uuid4())}'
    s3 = boto3.resource('s3')
    region = sagemaker.Session().boto_region_name
    bucket_config = {'LocationConstraint': region}
    s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=bucket_config)

    with open('pickled_data/pytorch_bucket_name.pickle', 'wb') as f:
        pickle.dump(bucket_name, f)
    print('Bucket Name:', bucket_name)


s3_uploader = sagemaker.s3.S3Uploader()

for s in splits:
    data_s3_uri = s3_uploader.upload(
        local_path     = (resized_path / s).as_posix(),
        desired_s3_uri = f's3://{bucket_name}/data/{s}')