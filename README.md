---
# Advanced Topics in Machine Learning
### Course at UCL (University College London) by Google DeepMind  
#### COMPGI13 Assignment 3 - Reinforcement Learning - Train Atari games using OpenAI gym
---
#### Nitish Mutha

##### email: nitish.mutha.16@ucl.ac.uk 
--- 


For each task I have added a command line API, so you can choose to run any model in either TRAIN or TEST mode. You can also select to run a model with specific configurations. In test mode code will pick models automatically based on arguments passed. For each task following are the API avaliable to execute the code from command line  

#### Dependencies
* Python 3.5.2
* TensorFlow version 1.0  
* gym  
* sklearn  
* numpy  
* pandas  
* matplotlib  
* scikit-image  
* random



## Problem A  
### 1 (GPU trained)
**Code:** `code/problemA1.py`  

#### API to run the code from command line:  
Command: `python problemA1.py`  

**Example:**
`python problemA1.py`  

---

### 2 (GPU trained)
**Code:** `code/problemA2.py`  

#### API to run the code from command line:  
Command: `python problemA2.py`  

**Example:**
`python problemA2.py` 

---

### 3 (GPU trained)
**Code:** `code/problemA3.py`  

**Model folders:** (for all learning rates, models gets restores automatically based on arguments passed in commands)  
**Linear**  
1. `models/problemA3/linear/1e-05`  
2. `models/problemA3/linear/0.00001`  
3. `models/problemA3/linear/0.0001`  
4. `models/problemA3/linear/0.001`  
5. `models/problemA3/linear/0.01`  
6. `models/problemA3/linear/0.5`  

**Hidden layer**  
1. `models/problemA3/hidden/1e-05`  
2. `models/problemA3/hidden/0.00001`  
3. `models/problemA3/hidden/0.0001`  
4. `models/problemA3/hidden/0.001`  
5. `models/problemA3/hidden/0.01`  
6. `models/problemA3/hidden/0.5`  

#### API to run the code from command line:  
Command: `python problemA3.py <arg1> <arg2> <arg3> <arg4>`  
Where,   
arg1 = (str) mode: e.g. `train`, `test`  
arg2 = (str) type: e.g. `linear`, `hidden`  
arg3 = (int) batch size: e.g. `32` (only for train mode)  
arg4 = (int) epochs: e.g. `50` or `100` (only for train mode)   

**Example:**
`python problemA3.py train linear 32 50`  
`python problemA3.py test linear`    

--- 

### 4 (GPU trained)
**Code:** `code/problemA4.py`  

**Model folders:** (models gets restores automatically based on arguments passed in commands)  
1. `models/problemA4/`  

#### API to run the code from command line:  
Command: `python problemA4.py <arg1> <arg2>`   
Where,   
arg1 = (str) mode: e.g. `train`, `test`  
arg2 = (float) learning rate: e.g. `0.0001`

**Example:**
`python problemA4.py test 32 50`    

--- 

### 5 (GPU trained)
**Code:** `code/problemA5.py`  

**Model folders:** (models gets restores automatically based on arguments passed in commands)  
1. `models/problemA5/1000`  
2. `models/problemA5/30`  

#### API to run the code from command line:  
Command: `python problemA5.py <arg1> <arg2> <arg3>`   
Where,   
arg1 = (str) mode: e.g. `train`, `test`  
arg2 = (int) hidden layer e.g. `30` or `1000`  
arg3 = (float) learning rate: e.g. `0.0001`

**Example:**
`python problemA5.py test 1000 0.0001`      

---

### 6 (GPU trained)
**Code:** `code/problemA6.py`  

**Model folders:** (models gets restores automatically based on arguments passed in commands)  
1. `models/problemA6`  

#### API to run the code from command line:  
Command: `python problemA6.py <arg1> <arg2> <arg3>`   
Where,   
arg1 = (str) mode: e.g. `train`, `test`  
arg2 = (int) batch size: e.g. `32` (only for train mode)  
arg3 = (float) learning rate: e.g. `0.0001`

**Example:**
`python problemA6.py test 32 0.0001`  

---

### 7 (GPU trained)
**Code:** `code/problemA7.py`  

**Model folders:** (models gets restores automatically based on arguments passed in commands)  
1. `models/problemA7`  

#### API to run the code from command line:  
Command: `python problemA7.py <arg1> <arg2> <arg3>`   
Where,   
arg1 = (str) mode: e.g. `train`, `test`  
arg2 = (int) batch size: e.g. `32` (only for train mode)  
arg3 = (float) learning rate: e.g. `0.0001`

**Example:**
`python problemA7.py test 32 0.0001`  

---

### 8 (GPU trained)
**Code:** `code/problemA8.py`  

**Model folders:** (models gets restores automatically based on arguments passed in commands)  
1. `models/problemA8`  

#### API to run the code from command line:  
Command: `python problemA8.py <arg1> <arg2> <arg3>`   
Where,   
arg1 = (str) mode: e.g. `train`, `test`  
arg2 = (int) batch size: e.g. `32` (only for train mode)  
arg3 = (float) learning rate: e.g. `0.0001`

**Example:**
`python problemA8.py test 32 0.0001`  


## Problem B  (CPU trained)
**Code:** `code/problemB-targetnet.py`  

**Models folders:** (Trained with GRU only)  
1. Pong-v3: `models/Pong-v3`  
2. Boxing-v3: `models/Boxing-v3`  
3. MsPacman-v3: `models/MsPacman-v3`    
  

#### API to run the code from command line:  
Command: `python problemB-targetnet.py <arg1> <arg2> <arg3> <arg4> <arg5>`  
Where,  
arg1 = (str) agent: e.g. `pong`, `boxing` or  `pacman`    
arg1 = (int) question: e.g. `1`, `2` or `3`  
arg3 = (str) mode: e.g. `train`, `test`  
arg4 = (int) batch size: e.g. `32` (only for train mode)  
arg5 = (float) learning rate: e.g. `0.0001`

**Example:**
`python problemB-targetnet.py pong 3 test 32 0.0001`

---


### Setup to run source code  
1. Install TensorFlow on Anaconda environment (gpu version prefered for speed of execution), [setup for windows](https://nitishmutha.github.io/tensorflow/2017/01/22/TensorFlow-with-gpu-for-windows.html)
2. Install numpy, sklearn, matplotlib if not installed by default.  
3. Activate tensforflow environment. e.g. `activate tensorflow-gpu`
4. Navigate to source code directory and run each python file. (command line API prefered)

 

**P.S. The saved parameters have been trained using tensorflow version APIr1.0**
