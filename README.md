# OnTheEstimationOfProductionLossesInOnGridPhotovoltaicInstallationsResultingFromGridFaults

##Initialization
At the very beginning the python venv must be initialized and packages listed in requirements.txt needs to be installed.

##Dataset preparation

First of all the dataset file `datasets/orgdataset.tar.xz` need to be unpacked to `datasets/orgdataset.csv`. This file is an original dataset that contains raw data. Those data needs to be cleared and 
transform before processing. To achieve that use the command below.  
```
    (.venv):~/$python datasets/process_data.py
```
It will transform original dataset into its clear version. New file will appear inside `datasets/dataset.csv`

##Article images

Run script to plot article images:
```
    (.venv):~/$python examples/problem_analysis_paper_images.py
```

## Experimental run
Use commands below to evaluate different models with differential evolution. Each script will work until it will be terminated.
```
    (.venv):~/$python examples/problem_analysis_differencial_model_optimization.py -m seaippf
    (.venv):~/$python examples/problem_analysis_differencial_model_optimization.py -m nlastperiod
```
Above commands will generate ```~/cm/``` directory and place 2 folders inside, one folder for each run of
```problem_analysis_differencial_model_optimization.py```. Structure should look like this tree:

```
    cm/
    ├── 20241121-114328_db0_seaippf
    │         ├── images_0.pdf
    │         ├── images_116.pdf
    │         ├── images_148.pdf
    │         ├── images_190.pdf
    │         ├── images_194.pdf
    │         ├── images_20.pdf
    │         ├── images_308.pdf
    │         ├── images_42.pdf
    │         ├── images_44.pdf
    │         └── result.csv
    └── 20241126-072927_db0_nlastperiods
              ├── images_0.pdf
              ├── images_4.pdf
              ├── images_5.pdf
              └── result.csv
 ```
after generation files can be analyzed by either file analyser
```
    (.venv):~/$python  problem_analysis_output_result_file_processing.py
    INFO:__main__:Started
    INFO:__main__:~/OnTheEstimationOfProductionLossesInOnGridPhotovoltaicInstallationsResultingFromGridFaults/examples
    INFO:__main__: => 0. 20241121-114328_seaippf
    INFO:__main__: => 1. 20241126-072927_nlastperiods
    Enter file number: 1
    (...)
``` 
or model validation script
```
    (.venv):~/$python  problem_analysis_output_result_file_processing.py \
     -f cm/20241121-114328_seaippf cm/20241126-072927_nlastperiods \
     -m seaippf nlastperiods
```
link:
https://anonymous.4open.science/r/OnTheEstimationOfProductionLossesInOnGridPhotovoltaicInstallationsResultingFromGridFaults-06D5