# 442Server

This repository contains the python code for research project "Detective Echo" conducted during 2019 Fall, in CS442.

To use the codes, you have to seed the data collected from our [Android app](https://github.com/cjeon/442Client). 
Or you can test the codes with the sample data we provide. This data is the same data we used for final presentation and the final report.
[Google Drive link](https://drive.google.com/file/d/188HjHgrzjBbsEiA2rjSyb90xMyZcvRS-/view?usp=sharing).

# Project Structure
There are only two python files. **main.py** and **multiclass_amp_phase_exp.py**. Before mid presentation, we only had **main.py**,
and we thought we better split the code and clean the code up, so we added **multiclass_amp_phase_exp.py**. However, since our later codes
are in **multiclass_amp_phase_exp.py**, to replicate our results, you only need to see **multiclass_amp_phase_exp.py**.

In **main.py**, some basic functionalities such as **read_from_file()**, **fft_to_amplitude_band()** are defined.

In **multiclass_amp_phase_exp.py**, our data pipeline is defined. Once we set the data source (the .txt file), 
the data is read and preprocessed and used for classifier training. The code automatically prints out the train results.

# How to replicate the project results
First, clone this repository. Then, activate the virtual environment.

```
source venv/bin/activate
```
Then, install the required dependencies.

```
pip install -r requirements.txt
```

Then place the files in the project root directory. From now on, let's assume we have three text files, "cup.txt", "pen.txt", "pot.txt".

Then modify the code starting from line 68. This variable, named `classFileMap` contains mapping from `class name` to `file name`.

When set as below, 

```python
classFileMap = {
    "metal_container": "mc_all.txt",
    "heatgun": "hg_all.txt",
    "paper_cup": "pc_all.txt"
}
```

the program will load the file "mc_all.txt" from the root directory and use this data for class "metal_container".

For our example scenario, where there are three text files - "cup.txt", "pen.txt", "pot.txt", we could modify the code as below.

```python
classFileMap = {
    "cup": "cup.txt",
    "pen": "pen.txt",
    "pot": "pot.txt"
}
```

Then the program will associate "cup.txt" with class "cup", "pen.txt" with class "pen", "pot.txt" with class "pot".

Then if you run the script, you will automatically see the trained model result - the script does every other things for you!

# Explanations on each steps

If you are interested in the each steps the script takes to train the model, please read on.

First we load the text file specified in the `classFileMap` using function `to_raw_data_map`. This file contains raw FFT results.
We need to convert this data to magnitude or phase (or both) to use them as features.

So this raw data is converted to amplitude data using the function `to_amplitude_map`.

Then we perform PCA on the feature map to reduce the dimension size. (from line 100 to 103) 
Then we scale the data in order to maximize the performance of the classifier using `RobustScaler` (from line 119 to 121).

Then finally, we train classifiers. We tested 10 classifiers, 
but every other classifiers except for our chosen one (the Random Forest classifier) is commented out. (from line 123 to 134).

We train the classifiers and print the results in the for-loop in line 136 to 140.
