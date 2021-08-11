# Field Museum Botany / Grainger Center CNN Classification
#### _models_ directory

Contains the files used to define neural network layer architectures.

---

##### Class Hierarchy:
The CNNModel class (within `cnnmodel.py`) is an abstract template class which sets up a CNN architecture (layers), and 
compiles the model with standard hyperparameters.

CNNModel contains three abstract methods which must be implemented in a child class:
- `add_convolutional_layers()`
- `add_hidden_layers()`
- `add_output_layers()`

(Note: These methods should always be executed sequentially -- they are separated for human readability.)

---

## Directory layout

##### Files:


- **modeltrainingarguments.py**
    - Sets up, parses, and validate command-line arguments for model training.
- **model_training.py**
    - `ModelTrainer` helper class for training Keras models, coordinating between the executing training script,
      images on the local filesystem, and a NN model architecture (child class of `CNNModel`).
    - `CTCModelTrainer`, a child class of `ModelTrainer` with a different validation implementation at the end of each epoch.
- **cnnmodel.py**
    - Contains the abstract template class `CNNModel` -- see Class Hierarchy above.
- **smithsonian.py**
    - Contains the `SmithsonianModel` class, a child class of `CNNModel` based on the layer architecture as published in the paper 
        ["Applications of deep convolutional neural networks to digitized natural history collections," Schuettpelz, 
        Frandsen, et al. 2017](https://doi.org/10.3897/BDJ.5.e21139).
- **rnn_ctc.py** - *in active development, basic functionality*
    - Contains the `RnnCtc` class, a child class of `CNNModel` that implements a CNN-RNN-CTC architecture as proposed
      in ["Handwritten text recognition in historical 
      documents" Harald Scheidl, 2018](https://repositum.tuwien.at/handle/20.500.12708/5409).
- **transfer_learning_model.py** - *in paused development, not functional*