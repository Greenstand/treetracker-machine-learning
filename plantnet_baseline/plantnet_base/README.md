# PlantNet-300K

This repository contains the code used to produce the benchmark in the paper *"Pl@ntNet-300K: a plant image dataset with high label
ambiguity and a long-tailed distribution"*. You can find a link to the paper [here](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/7e7757b1e12abcb736ab9a754ffb617a-Paper-round2.pdf).
In order to train a model on the PlantNet-300K dataset, you first have to download the dataset [here](https://doi.org/10.5281/zenodo.4726653).

If you use this work for this research, please cite the paper :

    @inproceedings{garcin2021pl,
      title={Pl@ ntNet-300K: a plant image dataset with high label ambiguity and a long-tailed distribution},
      author={Garcin, Camille and Joly, Alexis and Bonnet, Pierre and Lombardo, Jean-Christophe and Affouard, Antoine and Chouet, Mathias and Servajean, Maximilien and Salmon, Joseph and Lorieul, Titouan},
      booktitle={NeurIPS 2021-35th Conference on Neural Information Processing Systems},
      year={2021}
    }

### Requirements

Only pytorch, torchvision are necessary for the code to run. 
If you have installed anaconda, you can run the following command :

```conda env create -f plantnet_300k_env.yml```

### Training a model

In order to train a model on the PlantNet-300K dataset, please first edit the hyperparameters.yaml file. 
Then, run the following command :

```python main.py```