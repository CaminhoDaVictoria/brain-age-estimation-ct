{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fnil\fcharset0 Verdana-Bold;\f1\fnil\fcharset0 Verdana;\f2\fnil\fcharset0 Verdana-Italic;
\f3\froman\fcharset0 TimesNewRomanPSMT;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;}
\paperw11900\paperh16840\margl1440\margr1440\vieww22900\viewh15140\viewkind0
\deftab720
\pard\pardeftab720\sl560\sa160\qj\partightenfactor0

\f0\b\fs40 \cf2 \expnd0\expndtw0\kerning0
Brain Age Estimation
\f1\b0\fs26 \cf0 \outl0\strokewidth0 \strokec2 \
\pard\pardeftab720\partightenfactor0
\cf0 \
This repository contains the code used for the analyses presented in the manuscript:\
\
\pard\pardeftab720\sl560\sa160\qj\partightenfactor0

\f2\i \cf2 \outl0\strokewidth0 Estimation of brain age from cranial Computed Tomography using 3D convolutional neural networks
\f1\i0 \cf0 \outl0\strokewidth0 \strokec2 \
\pard\pardeftab720\partightenfactor0
\cf0 \

\f0\b\fs36 Overview
\f1\b0\fs26 \
\
The repository provides scripts for data preprocessing, statistical analysis, model training, and figure generation.\
\

\f0\b\fs36 Repository Structure
\f1\b0\fs26 \
\
```text\
\uc0\u9500 \u9472 \u9472  scripts/\
\uc0\u9500 \u9472 \u9472  README.md\
\uc0\u9492 \u9472 \u9472  requirements.txt\
```\
\

\f0\b\fs36 Requirements
\f1\b0\fs26 \
\
The code was developed and tested using Python 3.14.4.\
\
Install the required packages with:\
\
```bash\
pip install -r requirements.txt\
```\
\

\f0\b\fs36 Usage
\f1\b0\fs26 \
\
Run the scripts in the following order:\
\
```bash\
python scripts/preprocessing.py\
python scripts/Med3D_DS1.py\
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 python scripts/Med3D_DS2.py\
python scripts/Med3D_DS3.py\
python scripts/Med3D_DS4.py\
python scripts/Med3D_DS5.py\
python scripts/Med3D_DS6.py\
python scripts/Med3D_DS7.py\
python scripts/Med3D_DS8.py\
python scripts/Med3D_DS9.py\outl0\strokewidth0 \strokec2 \
\pard\pardeftab720\partightenfactor0
\cf0 ```\
\

\f0\b\fs36 MedicalNet Pretrained Models
 \f1\b0\fs26 \
\
This project uses pretrained MedicalNet weights.
 
The pretrained checkpoints are not included in this repository and must be downloaded separately from the official MedicalNet repository.
 
After downloading, place the checkpoint files in:
 
MedicalNet_checkpoints/pretrain/
\

\f0\b\fs36 Data Availability
\f1\b0\fs26 \
\
The original data cannot be shared due to privacy and data protection restrictions.\
\

\f0\b\fs36 Reproducibility
\f1\b0\fs26 \
\
All analyses reported in the manuscript can be reproduced using the scripts provided in this repository, provided that access to the original data is available. Paths and labels must be updated accordingly\
\

\f0\b\fs36 Citation
\f1\b0\fs26 \
\
If you use this code, please cite:\
\

\f0\b\fs36 License
\f1\b0\fs26 \
\
This project is licensed under the MIT License.\
\

\f0\b\fs36 Contact
\f1\b0\fs26 \
\
Dr. Armin Bachhuber\
\pard\pardeftab720\partightenfactor0

\f2\i \cf2 \outl0\strokewidth0 Saarland University Hospital
\f1\i0  \
armin.bachhuber@uks.eu\cf0 \outl0\strokewidth0 \strokec2 \
\pard\pardeftab720\qc\partightenfactor0

\f3\fs29\fsmilli14667 \cf2 \outl0\strokewidth0 \
\pard\pardeftab720\partightenfactor0

\f1\fs26 \cf0 \outl0\strokewidth0 \strokec2 \
}