<h1 style="text-align:center;"> <u> BetterSquirrelDetector</u>🐿️ </h1>

Welcome to the BetterSquirrelDetector repository! This repository contains scripts, data, and models that accompany a series of blog posts on data-centric AI and active learning. It builds upon the original [SquirrelDetector](https://dagshub.com/yonomitt/SquirrelDetector) project.

---

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [Collaborators](#collaborators)

---
## Repository Structure

The repository is organized as follows:

- `.dvc`: Contains DVC files for tracking data and models.
- `.labelstudio`: Contains LabelStudio files for validation and test set images.
- `annotations`: Includes validation and test datasets.
- `data`: Data related to the project.
- `models`: Pretrained models for squirrel detection.
- `src`: Source code for various components of the project.

Inside the `src` folder:

- `data`: Scripts related to data handling and preparation.
- `webserver`: Code for the web server component.
- `Dockerfile`: Dockerfile for creating a containerized environment.
- `_wsgi.py`: Code for running a web server serving models from the MLflow Model Registry.
- `docker-compose.yml`: Configuration for Docker Compose.
- `get_or_create_mlflow_experiment.py`: Script for creating MLflow experiments.
- `ls_model_server.py`: Code for updating Label Studio API endpoint.
- `model_wrapper.py`: Wrapper for MLflow model registration.
- `register_model.py`: Script for registering the model wrapper.
- `train_squirrel_detector.py`: Script for training the squirrel detector.
- `upload_model.py`: Script for uploading models.

---
## Getting Started

To get started with this project, follow these steps:

1. Clone the repository:
```bash
git clone https://dagshub.com/yonomitt/BetterSquirrelDetector.git
```
2. To use MLflow Tracking: 
```bash
MLFLOW_TRACKING_URI=https://dagshub.com/yonomitt/BetterSquirrelDetector.mlflow \
MLFLOW_TRACKING_USERNAME=your_username \
MLFLOW_TRACKING_PASSWORD=your_token \
python script.py
```

## Contributing

Contributions to this project are welcome. To contribute, please follow the standard GitHub workflow:

1. Fork the repository.
2. Create a feature branch.
3. Make your changes.
4. Submit a pull request.

Please ensure your code adheres to the project's coding guidelines.

## Collaborators

- [Yono Mittlefehldt](https://dagshub.com/yonomitt)
- [Dean Pleban](https://dagshub.com/Dean)
- [Anna Hyatt](https://dagshub.com/anna)

---
Thank You for visiting the [BetterSquirrelDetector](https://dagshub.com/yonomitt/BetterSquirrelDetector) repository! We hope you find this project interesting and valuable. Happy coding! 💻🐿️
