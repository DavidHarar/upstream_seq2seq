# Summary


This folder contains the data, code and models for the upstream task in `project name to be defined`. The structure of the folder and its sub-folders is directly related to [Deep Learning in Production](https://leanpub.com/DLProd)'s chapter 4.

- configs: In this module, we define everything that can be configurable and can be changed in the future. Good examples are training hyperparameters, folder paths, metrics, flags, etc.
- dataloader: All the data loading and data pre-processing classes and functions live here.
- evaluation: Code files that aims to evaluate the performance and accuracy of our model.
- executor: in this folder, we usually have all the functions and scripts that train the model or use it for prediction in different environments. And by different environments, I mean executors for CPU-only systems, executors for GPUs, executors for distributed systems. This package is our connection with the outer world and it’s what our main.py will use.
- models: contains the actual deep learning code.
- notebooks include all our Jupyter/Colab notebooks in one place that we built in the experimentation phase of the machine learning lifecycle.
	- Here I deviete from this structure's and use [coockie cutter](https://github.com/bh1995/AF-classification/tree/cabb932d27c63ea493a97770f4b136c28397117f) suggestion on how to name notebook files: "Naming convention is a number (for ordering), the creator's initials, and a short `-` delimited description, e.g. `1.0-jqp-initial-data-exploration`."
- data: Data folder
	- raw: The original, immutable data dump.
	- processed: The final, canonical data sets for modeling.
