# Greenstand/PlantNet - Transfer Learning
This repo contain the source code needed to apply transfer learning from the PlantNet model to our Greenstand images.


### Requirements

Only pytorch, torchvision are necessary for the code to run. 
If you have installed anaconda, you can run the following command :

```conda env create -f plantnet_300k_env.yml```

### Training a model

In order to train a model, please first edit the hyperparameters.yaml file. 
Make sure you also have a pre-trained model and put it's location in the hyperparameters file. 
The program will also attempt to sync the data from s3 to your local store, as put in the local_path param. 

After this is all done, you can run the main.ipynb to begin training and testing.
