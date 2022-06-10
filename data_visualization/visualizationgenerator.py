import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from data_visualization.charts import ROCChart, AccuracyChart, LossChart, ConfusionMatrix
from tensorflow.keras.callbacks import History


class VisualizationGenerator:
    def __init__(self):
        self.folder_name = f'graphs{datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")}'
        assert not os.path.exists(self.folder_name), f'Error, the graph folder {self.folder_name} already exists!'
        os.makedirs(self.folder_name)
        self.all_charts = []
        self.all_charts.append(ROCChart(self.folder_name))
        self.all_charts.append(AccuracyChart(self.folder_name))
        self.all_charts.append(LossChart(self.folder_name))
        self.all_charts.append(ConfusionMatrix(self.folder_name))

    def update(self, validation_labels: np.array,
               prediction_probability: np.array,
               history: History,
               class_labels: list,
               predictions: np.array,
               current_class: int) -> None:
        for each in self.all_charts:
            each.update(validation_labels, prediction_probability, history, class_labels,
                        predictions, current_class)
            each.save(class_labels, current_class)

    def finalize(self) -> None:
        results = pd.DataFrame()
        # results['Fold'] = [1]
        for each in self.all_charts:
            each.finalize(results)
        results.to_csv(Path(self.folder_name, 'final_data.csv'), encoding='utf-8', index=False)
