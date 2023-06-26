import sys
sys.path.insert(0, 'C:\\University\\Drugandinteraction\\PRN_COM\\chemprop2')
from chemprop.args import TrainArgs, PredictArgs
from chemprop.train import cross_validate, run_training, make_predictions
import numpy as np
import pandas as pd

if __name__ == '__main__':

    # train data
    arguments = [
        '--data_path', 'data/P30542_same.csv',
        '--dataset_type', 'regression',
        '--save_dir', 'checkpoints/chemprop2_test_checkpoint',
        '--epochs', '5',
        '--smiles_columns', 'Drug',
        '--target_columns', 'regression_label',
        '--loss_function', 'mse',
        '--dropout', '0.2',
        '--sequence_columns', 'Target_Seq',
    ]

    args = TrainArgs().parse_args(arguments)
    mean_scores, std_scores = cross_validate(args, train_func=run_training)

    # test data
    arguments = [
        '--test_path', 'data/regression_test_smiles.csv',
        '--preds_path', 'results/chemprop2_test.csv',
        '--checkpoint_dir', 'checkpoints/chemprop2_test_checkpoint',
        '--smiles_columns', 'smiles',
    ]

    args = PredictArgs().parse_args(arguments)
    # model_objects = chemprop.train.load_model(args=args)
    # preds = make_predictions(args=args)