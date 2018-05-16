# Tensorflow-android-classifier
Descreption: There are many tutorial about tensorflow on android, but when i was a student at university, it hard to us learn and approach tensorflow, until i watch many video and read many article about it.

## Theory
TensorFlow is an open source library for numerical computation, specializing in machine learning applications. Tensorflow's developed by Google and who likes opensoure. So i'd like to use TF in this project to classifier objects.

### Tensorflow
* Tensorflow provide high level api and low level api to manipulation with data.
* TF support Multi-GPU
* Training across distributed resources (i.e., cloud).
* Queues for putting operations like data loading and preprocessing on the graph.
* Visualize the graph itself using TensorBoard.
* Logging events interactively with TensorBoard.
* Model checkpointing.
* Performance and GPU memory usage are similar to Theano and everything else that uses CUDNN.
* Get started with TF: https://www.tensorflow.org/get_started/

### Tensorflow for android
* Get started with Tensorflow Android: https://www.tensorflow.org/mobile/android_build
* Get more: https://bit.ly/2Ikuwnt (Slide for android developer)

## Created first app Tensorflow-android-classifier

### Prepare data: 
You need one dataset for tranning. If you want to classifier flower, flag, car,... you need 1 dataset of them. <br>
Guide:
* Fatkun downloader chrome extension: go google, type what you need, click tab images (there are many images), then click Fatkun logo, you NEED CLEAN the DATASET, delete all images not involve, download all of extractly you need. 

### Prepare environment:
To work with tensorflow and android, you need: 
* Android NDK:
* Android SDK:
* Tensorflow:
* Tensorflow broad (optional)
* Tranning script: retrain.py, optimize_for_inference.py, label_images.py, quantize_graph.py 

You should update the workspace and build bazeltool. It lost many time and can failed when run, you must run on computer > 16GB RAM. It sad :(

<b>Solution: </b>
  
Install and run docker (all in one) every you need at here:  

* Window (i suggest Docker toolbox)
* Linux (Docker CE)

When you downloaded, run this command to install docker container:

<b>docker run -it -p 8888:8888 -v $HOME/tf_files:/tf_files tensorflow/tensorflow:nightly-devel</b>

It share folder tf_files at your main computer with docker container

## Demo
