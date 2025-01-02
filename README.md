# KRANE

**This is the data and code for our paper** `Knowledge-Driven and Relation-Aware Synergistic Learning for Drug Repositioning`.

## Requirements
* `Python(version >= 3.6)`
* `pytorch(version>=1.4.0)`
*  `ordered_set(version>=3.1)`
* `pytorch>=1.7.1 & <=1.9`
* `numpy(version>=1.16.2)`
* `torch_scatter(version>=2.0.4)`
* `scikit_learn(version>=0.21.1)`
* `torch-geometric`
  

We highly recommend you to use conda for package management.

## Datasets

We provide the dataset in the [data](data/) folder.

- [GP-KG](data/GP-KG.zip)
- [OpenBioLink](data/Openbiolink.zip)
- [Biokg](data/Biokg.zip)

## Repository structure

The current repository is structured in the following way:
```
|-- README.md
|-- main.py
|-- test.py
|-- config
|   |-- log_config.json
|-- data (Data folder)
|   |-- GP-KG.zip
|   |-- Openbiolink.zip
|   `-- Biokg.zip
`-- model
    |-- RAFE.py
    |-- data_loader.py
    |-- message_passing.py
    |-- predict.py
    |-- tools.py
```

## Model

The basic structure of our model an be found in [model](model/) folder.
The model can be divided into 3 parts, Rational-Aware Feature Extractor, Synergistic Feature Reconstruction module and Knowledge-Regulated Loss. 

## Training
```
    python main.py -data dataset_directory -gpu 0 -name test_model -epoch 400
```

## Predicting
```
    python test.py -data test_data -gpu 0 -name test_model -test_file drug_pre.txt -save_result result.txt
```

