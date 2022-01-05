import os
import pandas as pd
from data_visualization.charts import ROCChart, AccuracyChart, LossChart, ConfusionMatrix


class VisualizationGenerator:
    def __init__(self, n_folds: int):
        self.folder_name = 'graphs'
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        self.all_charts = []
        self.all_charts.append(ROCChart(self.folder_name))
        self.all_charts.append(AccuracyChart(self.folder_name))
        self.all_charts.append(LossChart(self.folder_name))
        self.all_charts.append(ConfusionMatrix(self.folder_name))

        self.n_folds = n_folds

    def update(self, history, index, validation_labels, prediction_probability, class_labels, predictions,
               current_class) -> None:
        for each in self.all_charts:
            each.update(index, validation_labels, prediction_probability, history, class_labels, predictions,
                        current_class)
            each.save(index, class_labels, current_class)

    def finalize(self) -> None:
        results = pd.DataFrame()
        results['Fold'] = list(range(1, self.n_folds + 1))

        for each in self.all_charts:
            each.finalize(results)
        results.to_csv(os.path.join(self.folder_name, 'final_data.csv'), encoding='utf-8', index=False)
