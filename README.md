

## [Kaggle: Google Landmark Recognition 2020](https://www.kaggle.com/c/landmark-recognition-2020)

<!-- <img src="https://storage.googleapis.com/kaggle-competitions/kaggle/19231/logos/header.png"> -->

## How to run experiments

Assume that you successfully downloaded and unpacked data to `input` directory
with structure:

```
input
├── sample_submission.csv
├── test
├── train
├── train.csv
└── train_valid.pkl
```

To run experiment:

```bash
PYTHONPATH=. python3 src/experiments/<experiment>.py
```