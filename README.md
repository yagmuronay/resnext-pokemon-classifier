# ResNext Pokemon Classifier with PyTorch
Pokemon classifier using resnext50 model from PyTorch as the backbone.

This repository includes:

- A finetunded PyTorch resnext50 model for classifying Pokemons to the top 3 classes.
- Training and validation dataset of Pokemon images, classified into 18 classes.
- pokemon_classifier\_train  (.py or .ipynp): Training of the model. 
- pokemon_classifier\_test  (.py or .ipynp): Validating the model performance.

The model with the best parameters is going to be saved once the training is over.

## Getting Started

- Decide if you will use the notebook (.ipynb) or python (.py) files for training/ validation. If using .ipynp make sure you import the dataset and .ipynb files to your drive and make sure the path variables intialized correctly in .ipynp files.

- Choose your device - use GPU or not?

```
# Default settings
# device = 'cpu' # Uncomment if using CPU
device = 'cuda'  # Using GPU, comment if using CPU
```

- Optional: You can modify the path to save the models, as well as the path to pokemon dataset by changing the relevant code. You can also modify the name when saving the model (default='checkpoint.pth').
- Run .py file or import .ipynb file to your Google Colab Repository and run from there.
- Your models will be saved during the training. At the end you can find the best model in checkpoint.pth.

### Prerequisites

```
os
csv
tqdm
PIL
matplotlib
numpy
torch
torchvision
tensorflow
```
## Running the tests
- Analaog to the training phase  decide if you will use the notebook (.ipynb) or python (.py) file. If using .ipynp file, make sure you have imported the dataset and .ipynb file  to your drive for the validation. Make sure the path variables intialized correctly.

- Choose your device - use GPU or not?

```
# Default settings
# device = 'cpu' # Uncomment if using CPU
device = 'cuda'  # Using GPU, comment if using CPU
```
Run pokemon_classifier\_test.py / .ipynb.

## Authors
* **Yagmur Onay** - *Implementation on initial code* - [Yagmur Onay](https://github.com/yagmuronay)
* **Hyemin Ahn** - *Initial code* 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This project builds on some skeleton code from the Deep Learning Lecture held at the Technical University of Munich.

