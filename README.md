# Brain Age Estimation

This repository contains the code used for the analyses presented in the manuscript:

Estimation of brain age from cranial Computed Tomography using 3D convolutional neural networks

## Overview

The repository provides scripts for data preprocessing, statistical analysis, model training, and figure generation.

## Repository Structure


├── scripts/\
├── README.md\
└── requirements.txt


## Requirements

The code was developed and tested using Python 3.14.4.

Install the required packages with:


pip install -r requirements.txt


## Usage

Run the scripts in the following order:


python scripts/preprocessing.py\
python scripts/Med3D_DS1.py\
python scripts/Med3D_DS2.py\
python scripts/Med3D_DS3.py\
python scripts/Med3D_DS4.py\
python scripts/Med3D_DS5.py\
python scripts/Med3D_DS6.py\
python scripts/Med3D_DS7.py\
python scripts/Med3D_DS8.py\
python scripts/Med3D_DS9.py


## MedicalNet Pretrained Models

This project uses pretrained MedicalNet weights. The pretrained checkpoints are not included in this repository and must be downloaded separately from the official MedicalNet repository. After downloading, place the checkpoint files in:Â MedicalNet_checkpoints/pretrain/

## Data Availability

The original data cannot be shared due to privacy and data protection restrictions.

## Reproducibility

All analyses reported in the manuscript can be reproduced using the scripts provided in this repository, provided that access to the original data is available. Paths and labels must be updated accordingly

## Citation

If you use this code, please cite:

CaminhoDaVictoria. (2026). CaminhoDaVictoria/brain-age-estimation-ct: Initial public release (Version v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23066012

## License

This project is licensed under the MIT License.

## Contact

Dr. Armin Bachhuber\
University Hospital of the Saarland \
armin.bachhuber@uks.eu


