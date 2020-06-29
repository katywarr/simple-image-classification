# Getting Started

## Clone this repository

Navigate to the folder that you would like to store this repository in and clone the repository.

```
git clone git@github.com:katywarr/simple-image-classification.git
```
To keep up-to-date with any changes to the repository:

```
cd simple-image-classification
git pull
```

# Setting up your environment with Anaconda

The Anaconda package contains python and many of the required dependencies for this project.
Instructions for downloading it are [here](https://docs.anaconda.com/anaconda/install/)

In recent versions of Anaconda, it is recommended that you do *not* select the option to add 
the Anaconda bin folder to your PATH during installation but use the Anaconda Prompt.

## Using the Anaconda Prompt 

On windows: Click start and start typing "Anaconda" to get the prompt.

## If your machine does not have a GPU

If your machine does not have a GPU, edit the `simple-image-classification.yml` to remove the TensorFlow GPU dependencies and use non-GPU TensorFlow.

## Create a virtual Python environment (one-time)

From within an Anaconda command prompt, navigate to the `simple-image-classification` folder and
create a virtual environment using the following command: 

*You only need to do this once.*

```
conda env create -f simple-image-classification.yml 
```

Whenever you want to use this environment, invoke:

```
conda activate simple-image-classification
```

Your prompt should now look like something like this:

```
(simple-image-classification) current_dir>
```

Create a Kernel to enable use of the conda environment in Jupyter 

```
 python -m ipykernel install --user --name simple-image-classification --display-name "Python (simple-image-classification)"
```


