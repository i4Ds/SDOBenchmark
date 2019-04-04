# Introduction
Please study the report under ‘docs/report/SFP@FHNW Report.pdf’.

# Installation

1. Install python 3.6 (64-bit)
2. Install virtualenv:

    On macOS and Linux:  
    ```python3 -m pip install --user virtualenv```
    
    On Windows:  
    ```py -m pip install --user virtualenv```

3. Clone the repository

4. Download the Dataset

    * Go to: https://i4ds.github.io/SDOBenchmark/#
    * Download "example ZIP" and unpack into "data/sample/"
    * Download "full ZIP" and unpack into "data/full/"

5. Create a virtualenv
    
    On macOS and Linux:  
    ```python3 -m virtualenv env```
    
    Windows:  
    ```py -m virtualenv env```

6. Activate the virtualenv
    
    On macOS and Linux:  
    ```source env/bin/activate```
    
    On Windows:  
    ```.\env\Scripts\activate```

7. Install packages from requirements.txt file:
 
    ```pip install -r requirements.txt```

    A note on Tensorflow:  
    The `requirements.txt` file contains the tensorflow
    package for CPU-only. See [Install Tensorflow](https://www.tensorflow.org/install/)
    for other available packages, such as tensorflow-gpu, or when problems with
    the installation of tensorflow arise.

# Train & Evaluate Models
To train and evaluate a model, run the respective model script:  
```python <modelscript.py>```

Evaluation logs are saved to the following folder: `logs/<model_name>/<date>`.

Run tensorboard, in order to view and compare evaluation metrics:  
```tensorboard --logdir=<logs directory>```