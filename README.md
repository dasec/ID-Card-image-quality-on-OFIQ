# ID-Card image quality assessment

This repository contains python code to assess the quality of ID Card images. 
It contains a preprocessing pipeline and various quality metrics.

For detailed information, please refer to the [usage example](example_usage.ipynb).

This repository contains the code referenced by the Paper "Image Quality Assessment of Identity Cards Using Measures from Open Face Image Quality"

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

### Citation

```bibtex
@inproceedings{grote2026idiqa,
  title     = {Image Quality Assessment of Identity Cards Using Measures from Open Face Image Quality},
  author    = {Grote, Gregor and Tapia, Juan E. and Rathgeb, Christian},
  booktitle = {Proceedings of the International Workshop on Biometrics and Forensics (IWBF)},
  year      = {2026},
  address   = {Darmstadt, Germany},
  institution = {Darmstadt University of Applied Sciences},
  doi = {xxx},
  pages = {xxx}
}
```

### Disclaimer

This repository is provided for research purposes only. The code and materials are shared to support academic study and reproducibility.

The datasets used in this analysis are not distributed through this repository. They must be requested directly from their respective original sources and are subject to their individual licenses and usage restrictions.

If you have any questions regarding this repository or the research, please contact: [gregor.grote@h-da.de](mailto:gregor.grote@h-da.de).

