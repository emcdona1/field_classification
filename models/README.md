# Convolutional Neural Networks and Microplants
#### _models_ directory

Contains the files used to define neural network layer architectures.

---

##### Class Hierarchy:
The CNNModel class (within `cnnmodel.py`) is an abstract template class which sets up a CNN architecture (layers), and compiles the model with standard hyperparameters.

CNNModel contains three abstract methods which must be implemented in a child class:
- `add_convolutional_layers()`
- `add_hidden_layers()`
- `add_output_layers()`

(Note: These methods are always executed sequentially -- they are separated into 3 methods solely for human readability.)

---

## Directory layout

##### Files:

- **cnnmodel.py**
    - Contains the abstract template class -- see Class Hierarchy above.
- **smthsonian.py**
    - Contains an implementation of the template class based on the layer architecture as published in the paper ["Applications of deep convolutional neural networks
to digitized natural history collections," Schuettpelz, Frandsen, et al. 2017](https://doi.org/10.3897/BDJ.5.e21139).