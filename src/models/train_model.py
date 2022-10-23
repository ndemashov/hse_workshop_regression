# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import r2_score
from src.utils import save_as_pickle
import pandas as pd
import catboost as cb
import src.config as cfg
import os

@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_target_filepath', type=click.Path(exists=True))
@click.argument('output_model_filepath', type=click.Path())
@click.argument('output_validx_filepath', type=click.Path())
def main(input_data_filepath, input_target_filepath, output_model_filepath, output_validx_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train = pd.read_pickle(input_data_filepath)
    target = pd.read_pickle(input_target_filepath)


    RANDOM_STATE = 77
    N_SPLITS = 5

    train_idx, val_idx = train_test_split(
        train_data.index, 
        test_size=0.8, 
        random_state=RANDOM_STATE
    )
    
    train_data, val_data = train.loc[train_idx], train.loc[val_idx]
    train_target, val_target = target.loc[train_idx], target.loc[val_idx]
    
    kf = KFold(
        n_splits=N_SPLITS, 
        shuffle=True, 
        random_state=RANDOM_STATE
    )

    for train_idx, test_idx in kf.split(train):
        X_train, X_test = train.iloc[train_idx], target.iloc[test_idx]
        y_train, y_test = train.iloc[train_idx], target.iloc[test_idx]
        
        ridge = Ridge(random_state=RANDOM_STATE).fit(X_train, y_train)

    parameters = {
        'fit_intercept': [True, False],
        'alpha': [0.1, 1, 5, 10, 15, 100],
        'tol': [1e-5, 1e-3, 1e-1],
        'positive': [True, False]
    }    
    model = Ridge(random_state=RANDOM_STATE, max_iter=1000)
    clf = GridSearchCV(model, parameters, scoring='r2', cv=3)
    clf.fit(train_data, train_target)
    clf.best_params_


    ridge = Ridge(random_state=RS, max_iter=1000, **clf.best_params_).fit(train_data, train_target)
    

    ridge.save_model(os.path.join(output_model_filepath, "ridge.cbm"))

    pd.DataFrame({'indexes':val_idx.values}).to_csv(output_validx_filepath)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
