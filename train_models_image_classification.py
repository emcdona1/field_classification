import random
import numpy as np
import matplotlib
import tensorflow as tf
from labeled_images.labeledimages import LabeledImages
from models.smithsonian import SmithsonianModel
from models.model_training import ModelTrainer
from utilities.timer import Timer
from models.modeltrainingarguments import ModelTrainingArguments

matplotlib.use('Agg')  # required when running on server


def main() -> None:
    timer = Timer('Model training')

    cnn_arguments = ModelTrainingArguments()
    new_images = LabeledImages(SEED)
    new_images.load_training_images(cnn_arguments.training_image_folder, cnn_arguments.image_size,
                                    cnn_arguments.color_mode, shuffle=True)
    architecture = SmithsonianModel(SEED, cnn_arguments.lr, cnn_arguments.image_size, cnn_arguments.num_output_classes,
                                    cnn_arguments.color_mode)

    trainer = ModelTrainer(cnn_arguments.n_epochs, cnn_arguments.batch_size, architecture, SEED)
    trainer.train_and_save_all_models(new_images)
    trainer.charts.finalize()

    for idx, class_label in enumerate(new_images.class_labels):
        print(f'Class {idx}: {class_label}')
    timer.stop()
    timer.print_results()


if __name__ == '__main__':
    SEED = 1
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    random.seed(SEED)

    main()
