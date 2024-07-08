# IMS_GANs
Experiments for GANs on IMS

# How to use
The data is not included in this git.
To use a custom dataset put an train-test split .npy file in the data folder.
Set the --datafile argument to reference the custom file

run the main.py file to execute an experiment.

# Contains 
main.py -> running this file executes an experiment, see argument parser for parameters
example_schedular.py -> this file is an example of how many experiments were run, summarized this file executes the main.py file with set arguments
models.py -> this file contains different generator and discriminator models
utils.py -> this file contains the custom functions needed for executing an experiment

discriminator_as_classifier.ipynb -> is used to run the discriminator design tests
simplefunctionestimator.ipynb -> is used to run the generator design tests

