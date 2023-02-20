# Alternative Regression Losses in Deep-Learning
Master Thesis. Studying the effect of  alternative regression losses in feedforward neural networks.


## About
nn_script is a python script that automatizes the study of diferent Feedforward Neural Nerworks configurations, designed to be used in supercomputers.
Some examples can be found at Results folder.

## Usage 
The script's call can be configured like this
```python
./nn_script  -r,--report_name report <str> 
                  -n,--n_jobs <int> 
                  -d,--datasets <'all' | comma separated names> 
                  -l,--losses <'all' | comma separated names> 
                  -v,--debug <bool> 
                  -s,--size comma-separated-ints 
                  -e,--epochs <int>  
                  -t,--tolerance <int> (without this option training is fixed for --epochs epochs) 
                  -p,--patience <int> (without this option training is fixed for --epochs epochs) 
                  -b,--batch_size <int>
 ```

Example:
```python
./nn_script --report_name report  --n_jobs 5 --datasets boston_housing --losses mse --size 20,20 --epochs 5000  --tolerance 1.e-16 --patience 500 --batch_size 200 --debug True
 ```

Evolution of the script is shown by default in the standard output:
![alt text](https://github.com/adriwitek/-Alternative-Regression-Losses-in-Deep-Learning/blob/main/img/execution_screenshot_1.png "Terminal Output Screenshot")

This is an example of how a report of the script looks like:
![alt text](https://github.com/adriwitek/-Alternative-Regression-Losses-in-Deep-Learning/blob/main/img/execution_screenshot_2.png "Terminal Output Screenshot2")



## Data Flow within each neural network

This is how Keras and Scikit-learn are connected. This diagram shows a complete meta-estimator. It's done this way so it is possible to do things easier,  like a Grid CV Search.

![alt text](https://github.com/adriwitek/-Alternative-Regression-Losses-in-Deep-Learning/blob/main/img/meta_estimator.png "Data Flow in each neural model.")


## MACROS 
Some macros are configurable. See code for further details.


## Requeriments
Important since different package versions with some libraries may cause some problems!

| Library  | Recommended Version |
|:--------------------------------------------------------------:|:-------:|
| Python                                                          |3.9 (or at least 3.8)|
| tensorflow                    |2.9.2| 
| keras                  |2.9.0| 
| Scikit-Learn                  |1.1.3| 
| Scikeras                  |0.9.0| 
| Matplotlib                  |3.5.2| 
| Pandas                  |1.4.4| 
| NumPy                 |1.22.3| 


                 
## Installation



I recommend creating first an enviroment like conda to avoid package version problems:
```python
 conda create --name nn_env
 conda activate nn_env
 ```
And then install the dependencies:
```python
 pip install -r requirements.txt 
 ```
 


## Applying Scikeras patch

 **IMPORTANT!**
 Some losses (custom implemented ones like epsilon-insensitive) require patching scikeras installation by replacing a file, due to keras - scikeras implementation. 
 This patch can be foung at **"/scikeras_patch folder/_saving_utils.py"** and should replace the **"__init__.py"** file in scikeras installation folder.
 
CAUTION: It **may need superuser rights**, so in order to avoid it you can use an execution environment like conda.

You can path the scikeras lib by like this:


```python
mv scikeras_patch/_saving_utils.py  <<YOUR PYTHON INSTALLATION PATH>>/lib/<<YOUR PYTHON version>>/site-packages/scikeras/__init__.py
 ```

For example using a miniconda installation in home folder with python 3.9 command became:

```python
mv scikeras_patch/_saving_utils.py  /home/adriwitek/miniconda3/lib/python3.9/site-packages/scikeras/__init__.py
 ```