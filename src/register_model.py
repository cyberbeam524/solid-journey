import argparse
import cloudpickle
import mlflow
import os
import torch

from sys import version_info

from get_or_create_mlflow_experiment import get_experiment_id

import model_wrapper
from model_wrapper import SquirrelDetectorWrapper

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

PYTHON_VERSION = "{major}.{minor}.1".format(major=version_info.major,
                                            minor=version_info.minor)

conda_env = {
    'channels': ['defaults'],
    'dependencies': [
        'python~={}'.format(PYTHON_VERSION),
        'pip',
          {
            'pip': [
                'mlflow',
                'pillow',
                'cloudpickle=={}'.format(cloudpickle.__version__),
                'torch>=1.12.0'
            ],
          },
    ],
    'name': 'squirrel_env'
}

def main():
    parser = argparse.ArgumentParser('Creates/gets an MLflow experiment and registers a YOLOv5 model to the Model Registry')
    parser.add_argument('--name', help='MLflow experiment name')
    parser.add_argument('--model', help='Path to saved YOLOv5 PyTorch model')
    parser.add_argument('--model-name', help='Registered model name')

    args = parser.parse_args()

    artifacts = { 'path': args.model }

    model = SquirrelDetectorWrapper()

    exp_id = get_experiment_id(args.name)

    cloudpickle.register_pickle_by_value(model_wrapper)

    with mlflow.start_run(experiment_id=exp_id):
        mlflow.pyfunc.log_model(
            'finetuned',
            python_model=model,
            conda_env=conda_env,
            artifacts=artifacts,
            registered_model_name=args.model_name
        )


if __name__ == '__main__':
    main()
