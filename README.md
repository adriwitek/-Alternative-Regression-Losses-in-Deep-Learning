# -Alternative-Regression-Losses-in-Deep-Learning
Master Thesis. Study the effect of  Alternative Regression Losses in Deep Learning.


## About
nn_script is a python script that automatizes the study of diferent Feedforward Neural Nerworks configurations, designed to be used in supercomputers.
Some examples can be found at Results folder.

## Usage 
Teh script call can be configured like this
```python
./nn_script --report_name report -r,--report_name <str> \n\t -n,--n_jobs <int> \n\t -d,--datasets <'all' | comma separated names> 
                \n\t -l,--losses <'all' | comma separated names> 
                \n\t -v,--debug <bool> 
                \n\t -s,--size comma-separated-ints 
                \n\t -e,--epochs <int>  
                \n\t -t,--tolerance <int> (without this option training is fixed for --epochs epochs) 
                \n\t -p,--patience <int> (without this option training is fixed for --epochs epochs) 
                \n\t -b,--batch_size <int>
 ```

Example
```python
./nn_script --report_name report  --n_jobs 5 --datasets boston_housing --losses mse --size 20,20 --epochs 5000  --tolerance 1.e-16 --patience 500 --batch_size 200 --debug True
 ```

## MACROS 
Some macros in the script are configurable. See code for further details.


## Requeriments
Versions are important, since different packege versions in some libraries cause some problems!

| Library  | Recommended Version |
|:--------------------------------------------------------------:|:-------:|
| Python                                                          |3.9 (or at least 3.8)|
| tensorflow                  || 
| keras                  || 
| Scikit-Learn                  |1.1.3| 
| Scikeras                  || 
| Matplotlib                  || 
| Pandas                  || 
| NumPy                 || 


                 
## Installation

 **IMPORTANT!**
 Some losses require patching scikeras instalation, replacing a file due to keras - scikeras implementation. This patch can be foung at /scikeras_patch folder.
 It may need superuser rights, so in order to avoid it you can use an execution environment like conda.

I recommend creating first an enviroment like conda to avoid package version problems:
```python
 conda create --name nn_env
 conda activate nn_env
 ```
And then install the dependencies:
```python
 pip install -r requirements.txt 
 ```
 
