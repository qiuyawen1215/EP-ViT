## Overview

This project is the implementation code of the paper "EP-ViT: Leveraging Encoding Priors to Eliminate Spatio-Temporal Redundancy in Vision Transformer for Efficient Inference" (under review) submitted to ECAI 2025.


## Dependencies

Dependencies are managed using Conda. The environment is defined in `environment.yml`.

To create the environment, run:
```
conda env create -f environment.yml
```

## Running Scripts

Scripts should be run from the repo's base directory.

Many scripts expect a `.yml` configuration file as a command-line argument. These configuration files are in `configs`. The structure of the `configs` folder is set to mirror the structure of the `scripts` folder. For example, to run the `base_672` evaluation for the ViTDet VID model:
```
./scripts/evaluate/vitdet_vid.py ./configs/evaluate/vitdet_vid/base_672.yml
```

## Data

The `datasets` folder defines PyTorch `Dataset` classes for Kinetics-400, VID, and EPIC-Kitchens.

The Kinetics-400 class will automatically download and prepare the dataset on first use.

VID requires a manual download. Download `vid_data.tar` from [here](https://drive.google.com/drive/folders/1tNtIOYlCIlzb2d_fCsIbmjgIETd-xzW-) and place it at `./data/vid/data.tar`. The VID class will take care of unpacking and preparing the data on first use.

EPIC-Kitchens also requires a manual download. Download the videos from [here](https://drive.google.com/drive/folders/1OKJpgSKR1QnWa2tMMafknLF-CpEaxDbY) and place them in `./data/epic_kitchens/videos`. Download the labels `EPIC_100_train.csv` and `EPIC_100_validation.csv` from [here](https://github.com/epic-kitchens/epic-kitchens-100-annotations) and place them in `./data/epic_kitchens`. The EPICKitchens class will prepare the data on first use.


