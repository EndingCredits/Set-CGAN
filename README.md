# Set-condiioned DC-GAN

This is an adaptation of the standard class-conditioned DC-GAN so now generator (and discriminator) are conditioned on an additional example set (as opposed to an explicit class label) whose distribution it must match.

For a similar idea, but in an autoencoder set-up, see [this paper](https://arxiv.org/abs/1606.02185).


## Acknowledgements

This repository is based off the popular [DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow). Many thanks to Taehoon Kim / [@carpedm20](http://carpedm20.github.io/).

It also uses a GAN regulariser given [here](https://github.com/rothk/Stabilizing_GANs). Many thanks to the authors.


## Prerequisites

- Python 2.7 or Python 3.3+
- [Tensorflow 0.12.1](https://github.com/tensorflow/tensorflow/tree/r0.12)
- [SciPy](http://www.scipy.org/install.html)
- [pillow](https://github.com/python-pillow/Pillow)
- (Optional) [Align&Cropped Images.zip](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) : Large-scale CelebFaces Dataset
- (Optional) [matplotlib](https://matplotlib.org/) : Plotting tool for visualisation of synthetic shapes dataset


## Usage

Test on synthetic dataset:

    $ python main.py --dataset shapes --train
    
Or download dataset with:

    $ python download.py celebA

Train a model with downloaded dataset:

    $ python main.py --dataset celebA --use_tags --input_height=108 --train


You can use your own dataset (without central crop) by:

    $ mkdir data/DATASET_NAME
    $ mkdir data/DATASET_NAME/CLASS_1
    ... add images to data/DATASET_NAME/CLASS_1 ...
    ... add images to data/DATASET_NAME/CLASS_2 ...
                       ...
    ... add images to data/DATASET_NAME/CLASS_N ...
    $ python main.py --dataset DATASET_NAME --train
    $ python main.py --dataset DATASET_NAME
    $ # example
    $ python main.py --dataset=eyes --input_fname_pattern="*_cropped.png" --train
    
Alternatively, for each image you can create a .tags file of the same name with a list of tags separated by spaces.
    $ # example
    $ python main.py --dataset=eyes --use_tags


## Results

### celebA

5_o_clock_shadow:
![examples](assets/5_o_Clock_Shadow_examples.png)
![samples](assets/5_o_Clock_Shadow_samples.png)

bald:
![examples](assets/Bald_examples.png)
![samples](assets/Bald_samples.png)

bald:
![examples](assets/Wearing_Lipstick_examples.png)
![samples](assets/Wearing_Lipstick_samples.png)

### Anime faces
Dataset based on extracted faces from danbooru images tagged with 500 most popular characters, roughly 400 images per character

artoria_pendragon_(all):
![examples](assets/artoria_pendragon_(all)_examples.png)
![samples](assets/artoria_pendragon_(all)_samples.png)

louise_francoise_le_blanc_de_la_valliere:
![examples](assets/louise_francoise_le_blanc_de_la_valliere_examples.png)
![samples](assets/louise_francoise_le_blanc_de_la_valliere_samples.png)

urakaze_(kantai_collection):
![examples](assets/urakaze_(kantai_collection)_examples.png)
![samples](assets/urakaze_(kantai_collection)_samples.png)



