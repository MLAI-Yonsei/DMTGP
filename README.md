# COVID-19 Prediction with Doubly Multi-task Gaussian Process
### Gaussian Process Regression for COVID Epidemiology

This repository contains PyTorch implemenations of Doubly Multi-task Gaussian Process (DMTGP) Regression for COVID Epidemiology by Python.

## Introduction
This is a machine learning code that simultaneously forecasts the number of confirmed cases and deaths for each country using various information (The number of inoculation, Temperature, Humidity, Precipitation, COVID-19 variants information, Stringency Index, Population, Holidays Information). We conduct experiments with statistical model ARIMA, regression baselines (Linear Regression, Polynomial Regression, and Support Vector Regression), Single Task Gaussian Process (STGP) Regression, and Multi Task Gaussian Process (MTGP) Regression as baselines. 

### Project Tree
```
.
├── base
│   ├── arima.py
│   ├── baseline.py
│   ├── linear_regression.py
│   ├── polynomial_regression.py
│   └── svr.py
├── figures
│   ├── denmark_conf.png
│   ├── denmark_dead.png
│   ├── france_conf.png
│   ├── france_dead.png
│   ├── germany_conf.png
│   ├── germany_dead.png
│   ├── italy_conf.png
│   ├── italy_dead.png
│   ├── japan_conf.png
│   ├── japan_dead.png
│   ├── south_korea_conf.png
│   ├── south_korea_dead.png
│   ├── taiwan_conf.png
│   ├── taiwan_dead.png
│   ├── uk_conf.png
│   ├── uk_dead.png
│   ├── us_conf.png
│   └── us_dead.png
|
├── plot_results
│   └── ours
│       ├── SM+L-1-2020-02-27-0.005-3
│       │   └── south_korea
│       │       ├── conf
│       │       │   └── best-epoch.png
│       │       └── dead
│       │           └── best-epoch.png
│       ├── SML+P-1-2020-02-27-0.005-4
│       │   └── italy
│       │       ├── conf
│       │       │   └── best-epoch.png
│       │       └── dead
│       │           └── best-epoch.png
│       ├── SML+P-1-2020-02-27-0.01-4
│       │   └── japan
│       │       ├── conf
│       │       │   └── best-epoch.png
│       │       └── dead
│       │           └── best-epoch.png
│       ├── SML+P-2-2020-02-27-0.005-3
│       │   ├── uk
│       │   │   ├── conf
│       │   │   │   └── best-epoch.png
│       │   │   └── dead
│       │   │       └── best-epoch.png
│       │   └── us
│       │       ├── conf
│       │       │   └── best-epoch.png
│       │       └── dead
│       │           └── best-epoch.png
│       ├── SML+P-2-2020-02-27-0.05-4
│       │   └── denmark
│       │       ├── conf
│       │       │   └── best-epoch.png
│       │       └── dead
│       │           └── best-epoch.png
│       ├── SMP-1-2020-02-27-0.005-2
│       │   └── france
│       │       ├── conf
│       │       │   └── best-epoch.png
│       │       └── dead
│       │           └── best-epoch.png
│       ├── SMP-2-2020-02-27-0.005-3
│       │   └── germany
│       │       ├── conf
│       │       │   └── best-epoch.png
│       │       └── dead
│       │           └── best-epoch.png
│       └── SM+P-2-2020-02-27-0.1-4
│           └── taiwan
│               ├── conf
│               │   └── best-epoch.png
│               └── dead
│                   └── best-epoch.png
├── preprocessed_data
│   ├── denmark.csv
│   ├── france.csv
│   ├── germany.csv
│   ├── italy.csv
│   ├── japan.csv
│   ├── meta_denmark.csv
│   ├── meta_france.csv
│   ├── meta_germany.csv
│   ├── meta_italy.csv
│   ├── meta_japan.csv
│   ├── meta_south_korea.csv
│   ├── meta_taiwan.csv
│   ├── meta_uk.csv
│   ├── meta_us.csv
│   ├── south_korea.csv
│   ├── taiwan.csv
│   ├── uk.csv
│   └── us.csv
├── raw_data
│   ├── confirmation.csv
│   ├── dayoff.xlsx
│   ├── dead.csv
│   ├── inoculation.xlsx
│   ├── population.xlsx
│   ├── stringency_index.xlsx
│   ├── variation2.xlsx
│   ├── variation.xlsx
│   └── weather.xlsx
├── sh
│   ├── arima
│   │   └── arima.sh
│   ├── gp
│   │   ├── mtgp
│   │   │   ├── mtgp_denmark.sh
│   │   │   ├── mtgp_france.sh
│   │   │   ├── mtgp_germany.sh
│   │   │   ├── mtgp_italy.sh
│   │   │   ├── mtgp_japan.sh
│   │   │   ├── mtgp_south_korea.sh
│   │   │   ├── mtgp_taiwan.sh
│   │   │   ├── mtgp_uk.sh
│   │   │   └── mtgp_us.sh
│   │   └── stgp
│   │       ├── stgp_denmark.sh
│   │       ├── stgp_france.sh
│   │       ├── stgp_germany.sh
│   │       ├── stgp_italy.sh
│   │       ├── stgp_japan.sh
│   │       ├── stgp_south_korea.sh
│   │       ├── stgp_taiwan.sh
│   │       ├── stgp_uk.sh
│   │       └── stgp_us.sh
│   ├── linear
│   │   └── linear.sh
│   ├── polynomial
│   │   └── polynomial.sh
│   ├── svr
│   │   └── svr.sh
│   └── total.sh
├── baselines.py
├── data.py
├── metrics.py
├── gp_models.py
├── models.py
├── mtgp.py
├── paths.py
├── README.md
├── stgp.py
└── utils.py
```

## Implementation

We used the following Python packages for core development. We tested on `Python 3.19.16`.
```
pytorch                   2.0.0
gpytorch                  1.9.1
pandas                    2.0.0
numpy                     1.24.2
scikit-learn              1.2.2
scipy                     1.10.1
```

### Arg Parser

The script `dmtgp.py` allow to train and evaluate our proposed model.
The script `stgp.py` and `mtgp.py` allow to train and evaluate all gaussian process regression models.
The script `baselines.py` allows to train and evaluate all the baselines we consider.

To train proposed methods use this:
```
dmtgp.py (or stgp.py or mtgp.py)--                     \
        --seed=<SEED>                          \
        --nation=<NATION>                      \
        [--preprocess_data]                    \
        --start_date=<START_DATE>              \
        --obs_end_date=<OBS_END_DATE>          \
        --pred_end_date=<PRED_END_DATE>        \
        --num_variants=<NUM_VARIANTS>          \
        --model_type=<MODEL_TYPE>              \
        --num_mixture=<NUM_MIXTURE>            \
        --kernel_name=<KERNEL_NAME>            \
        --kernel_kr=<KERNEL_NAME>            \
        --kernel_jp=<KERNEL_NAME>            \
        --kernel_tai=<KERNEL_NAME>            \
        --rank=<RANK>                          \
        --max_epoch=<NUM_EPOCH>                \
        --lr_init=<LR_INIT>                    \
        --tolerance=<TOLERANCE>                \
        --eval_criterion=<EVAL_CRITERION>      \
        --eval_only                            \
        [--ignore_wandb]
```
Parameters:
* ```SEED``` &mdash; random seed (default: 1000)
* ```NATION``` &mdash; nation to train (default: all)
    - denmark
    - france
    - germany
    - italy
    - japan
    - south_korea
    - taiwan
    - uk
    - us
    - all (train all nations respectively)
* ```--preprocess_data``` &mdash; Preprocess data from raw data
* ```START_DATE``` &mdash; start date of data (default: 2020-02-27)
* ```OBS_END_DATE``` &mdash; last date of observed data. (default: 20203-02-08)
* ```PRED_END_DATE``` &mdash; last date of pred_only phase (default: 20203-12-31
* ```NUM_VARIANTS``` &mdash; the number of variants. (default: 28)
* ```MODEL_TYPE``` &mdash; model name (default: mtgp) :
    - ours (DMTGP)
    - MTGP
    - STGP
    - ARIMA
    - LINEAR
    - SVR
* ```NUM_MIXTURE``` &mdash; the number of mixture for spectral mixture kernel
* ```kernel_name``` &mdash; kernel name (default: SpectralMixture) :
    - RBF
    - SpectralMixture
    - SML (SpectralMixture * Linear)
    - SM+L (SpectralMixture + Linear)
    - SMP (SpectralMixture * Periodic)
    - SM+P (SpectralMixture + Periodic)
    - SMPL (SpectralMixture * Periodic * Linear)
    - SMP+L (SpectralMixture * Periodic + Linear)
    - SML+P (SpectralMixture * Linear + Periodic)
    - SM+P+L (SpectralMixture + Periodic + Linear)
* ```kernel_jp / kernel_tai / kernel_kr``` &mdash; specify kernel for each nation
* ```RANK``` &mdash; rank used for mtgp (default : 2)
* ```MAX_EPOCH``` &mdash; max epoch to train (default: 5000)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```TOLERANCE``` &mdash; tolerance for num epochs without improvement at validation set (default: 500)
* ```EVAL_CRITERION``` &mdash; evaluation criterion (default: MAE):
    - MAE
    - MLL
* ```eval_only``` &mdash; eval model without training
* ```--ignore_wandb``` &mdash; Ignore WandB to do not save results

----
### Preprocess Data
If uou wnat to preprocess raw data, you can use this.
```
sh sh/{model_type}/{bash_name}.sh --preprocess_data
```
The raw data consists of the following files, and the description for each file is as follows.

`confirmation.csv`
- This file contains information about daily confirmed cases by nation on separate sheets.

`dayoff.xlsx`
- This file contains information about holidays by nation.

`dead.csv`
- This file includes information about daily death counts by nation on separate sheets.


`inoculation.xlsx`
- This file contains information about the COVID-19 vaccination status by nation.

`population.xlsx`
- This file includes information about populations by nation.

`stringency_index.xlsx`
- This file contains information about the date-wise stringency index by nation.
- The data for the United Kingdom and the United States has been recorded at the state level, so we averaged for use.

`variation.xlsx`
- This file contains information about the types of variants recorded on a biweekly basis on separate sheets by nation.
- Currently, this file is not used for learning the model.

`variation2.xlsx`
- This file contains information about the most dominant COVID-19 variants recorded on a weekly basis on separate sheets by nation, along with the corresponding percentage it represents during that week.
- For nations not present in this file, data from nations within the same continent was initially utilized. Among those, information from the nearest country in terms of capital city distance was used as a substitute.

`weather.xlsx`
- This file contains information about daily temperatures, humidity, precipitation, and snowfall on separate sheets by nation.
- We have utilized temperature, humidity, and precipitation from this dataset.

While preprocessing the raw data, each piece of data is aggregated and stored in `./preprocessed_data` as training data for learning about a single nation.

### Train Models

If you want to train model or sweep model to search best hyperparameter, you can use this:

```
# baselines
sh sh/{baseline_name}/{baseline_name}.sh

# mtgp
sh sh/gp/mtgp/mtgp_{nation}.sh

# stgp
sh sh/gp/stgp/stgp_{nation}.sh
```

### plot results
If you run models, you can check plot results at `'./plot_results'`.

After training the data for each country using our model or baselines, the predicted results are as follows.
(If you want to generate plots in paper format, you can run `plot_test.ipynb`)

`Denmark`

<img src="./figures/denmark_conf.png" width="250"/> <img src="./figures/denmark_dead.png" width="250"/> 

`France`

<img src="./figures/france_conf.png" width="250"/> <img src="./figures/france_dead.png" width="250"/> 

`Germany`

<img src="./figures/germany_conf.png" width="250"/> <img src="./figures/germany_dead.png" width="250"/> 

`Italy`

<img src="./figures/italy_conf.png" width="250"/> <img src="./figures/italy_dead.png" width="250"/> 

`Japan`

<img src="./figures/japan_conf.png" width="250"/> <img src="./figures/japan_dead.png" width="250"/> 

`South Korea`

<img src="./figures/south_korea_conf.png" width="250"/> <img src="./figures/south_korea_dead.png" width="250"/> 

`Taiwan`

<img src="./figures/taiwan_conf.png" width="250"/> <img src="./figures/taiwan_dead.png" width="250"/> 

`United Kingdom`

<img src="./figures/uk_conf.png" width="250"/> <img src="./figures/uk_dead.png" width="250"/> 

`United States`

<img src="./figures/us_conf.png" width="250"/> <img src="./figures/us_dead.png" width="250"/> 