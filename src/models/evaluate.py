# load train dataset and val_idx
# filter by val_idx
# after look at https://github.com/iterative/example-get-started/blob/main/src/evaluate.py


import logging
import json

import click
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier

from sklearn.metrics import mean_squared_error 

@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_target_filepath', type=click.Path(exists=True))
@click.argument('input_model_filepath', type=click.Path(exists=True))
@click.argument('input_validx_filepath', type=click.Path(exists=True))

def main(input_data_filepath, input_target_filepath, input_model_filepath, input_validx_filepath):
  
    logger = logging.getLogger(__name__)
    logger.info('making validation metrics')

    train_data = pd.read_pickle(input_data_filepath)
    train_target = pd.read_pickle(input_target_filepath)

    val_indxes = pd.read_csv(input_validx_filepath)['indexes'].values
    
    val_data = train_data.loc[val_indxes]
    val_target = train_target.loc[val_indxes]

    trained_model = CatBoostClassifier().load_model(input_model_filepath)

    y_pred = trained_model.predict(val_data)

    metrics = {
        'rmse': np.sqrt(mean_squared_error(np.log(val_target), np.log(y_pred)))  
    }

    with open("reports/figures/metrics.json", "w") as outfile:
        json.dump(metrics, outfile)

main()
