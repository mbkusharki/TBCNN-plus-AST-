# TBCNN + AST--A Deep Learning model used because of its Hierarchical nature in Code Representation
This repository includes the code and MutantBench data in our paper entitled "Automatic Classification of Equivalent Mutnats in Mutation Testing of Android Application. We have applied our neural source code representation to source code classification. It is also expected to be helpful in more tasks. It is based on the need to classify equivalent mutants using a standard dataset.

### Requirements
+ python 3.6<br>
+ pandas 0.20.3<br>
+ gensim 3.5.0<br>
+ scikit-learn 0.19.1<br>
+ pytorch 1.0.0<br> (The version used in our paper is 0.3.1 and source code can be cloned by specifying the v1.0.0 tag if needed)
+ pycparser 2.18<br>
+ javalang 0.11.0<br>
+ Amazon SageMaker Studio Lab<br>
+ MutantBench<br>
+ Android Studio<br>
+ RAM 16GB or more
+ GPU with CUDA support is also needed
+ BATCH_SIZE should be configured based on the GPU memory size

### How to install
Install all the dependent packages via pip:

	$ pip install pandas==0.20.3 gensim==3.5.0 scikit-learn==0.19.1 pycparser==2.18 javalang==0.11.0
 
Install pytorch according to your environment, see https://pytorch.org/ 


### Source Code Classification
1. `cd tbcnn`
2. run `python pipeline.py` to generate preprocessed data.
3. run `python train_tbcnn.py` for training and evaluation
4. run *python test_tbcnn.py* for testing and performance evaluation matrix

### How to use it on your own dataset

Please refer to the `pkl` files in the corresponding directories of the two tasks. These files can be loaded by `pandas`.
 

}
