# Data

The datasets are **not included** in this repository. They are provided by Kaggle under each competition's rules. Download them and place them here so the folder looks like this:

```
data/
├── home-data-for-ml-course/            # notebook 01
│   ├── train.csv
│   └── test.csv
├── digit-recognizer/                   # notebook 02
│   ├── train.csv
│   └── test.csv
└── data-science-london-scikit-learn/   # notebook 03
    ├── train.csv
    ├── trainLabels.csv
    └── test.csv
```

## Option 1: Kaggle CLI

You need a Kaggle account, an API token (`kaggle.json`), and you must accept each competition's rules on its page first.

```bash
pip install kaggle
cd data

kaggle competitions download -c home-data-for-ml-course
kaggle competitions download -c digit-recognizer
kaggle competitions download -c data-science-london-scikit-learn

unzip home-data-for-ml-course.zip          -d home-data-for-ml-course
unzip digit-recognizer.zip                 -d digit-recognizer
unzip data-science-london-scikit-learn.zip -d data-science-london-scikit-learn
```

## Option 2: Manual download

Download the files from the **Data** tab of each competition:

| Notebook | Competition |
|----------|-------------|
| 01 | https://www.kaggle.com/competitions/home-data-for-ml-course |
| 02 | https://www.kaggle.com/competitions/digit-recognizer |
| 03 | https://www.kaggle.com/competitions/data-science-london-scikit-learn |
