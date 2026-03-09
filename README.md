# ID-Card image quality assessment

This repository contains python code to assess the quality of ID Card images. 
It contains a preprocessing pipeline and various quality metrics.

For detailed information, please refer to the [usage example](example_usage.ipynb).

### Installation

To create a virtual environment and install the required packages, run:

```bash
conda env create -f environment.yaml
conda activate oidiq
```

### Usage

To run the quality assessment on a set of images, use the `run.py` script. You can specify the input CSV file containing image paths, output CSV file for results, and other parameters. For example:

```bash
python run.py --i data/test_files.csv --o data/results.csv --batch-size 16
```
