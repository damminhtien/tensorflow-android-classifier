# Tensorflow-android-classifier
Keyword: machine-learning, deep-learning, classification, android-implement, computer-vision, convolutional neural network, tranfer-learning.

Description: There are many tutorial about tensorflow on android, but when i was a student at university, it hard to us learn and approach tensorflow, until i watch many video and read many article about it.

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
* Fatkun downloader chrome extension.

You NEED CLEAN the DATASET, delete all images not involve, download all of extractly you need. Copy all of images into 1 folder that's name is comom name of label of them.  

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

> docker run -it -p 8888:8888 -v $HOME/tf_files:/tf_files tensorflow/tensorflow:nightly-devel</b>

It share folder tf_files at your main computer with docker container. Update tensor:

> pip install --upgrade "tensorflow==1.7.*"

Your structure folder:

tf_files

---|-----script

---|-----photos

---|-----tf_files

tf_files at root of apps, script contains files .py that train and optimize the model, file photos contains folders of images input, tf_files contains result of next steps (files model, label, bottleneck,...)

## Train model
You need download script at:
* retrain.py: https://github.com/googlecodelabs/tensorflow-for-poets-2/blob/master/scripts/retrain.py
* optimize_for_inference.py: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py
* label_images.py: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/label_image.py
* quantize_graph.py: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/quantize/python/quantize_graph.py

Puts them into script folder in tf_files folder.

Investigate the retraining script
The retrain script is from the TensorFlow Hub repo, but it is not installed as part of the pip package. So for simplicity I've included it in the codelab repository. You can run the script using the python command. Take a minute to skim its "help".

> python -m scripts.retrain -h

Start your retraining with one big command (note the --summaries_dir option, sending training progress reports to the directory that tensorboard is monitoring) :

> python -m scripts.retrain \
>  --bottleneck_dir=tf_files/bottlenecks \
>  --how_many_training_steps=500 \
>  --model_dir=tf_files/models/ \
>  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
>  --output_graph=tf_files/retrained_graph.pb \
>  --output_labels=tf_files/retrained_labels.txt \
>  --architecture="${ARCHITECTURE}" \
>  --image_dir=tf_files/name_photos_folder

Note that: --how_many_training_steps=500, you can increase this number to get higher accuracy, ${ARCHITECTURE} you change it into inception_v3 (i use inceptionv3) or mobilenet, softmax,... --image_dir=tf_files/flower_photo, you must change this folder name by the folder name of photos you download at before step.

retrained_graph.pb & retrained_labels.txt is the most important file of this program.

Next, verify that the model is producing sane results before starting to modifying it.

The scripts/ directory contains a simple command line script, label_image.py, to test the network. Now we'll test:

>python -m scripts.label_image \
>  --graph=tf_files/retrained_graph.pb  \
>  --image=tf_files/photos/folder_name/photo_name.jpg

Optimize for inference: To avoid problems caused by unsupported training ops, the TensorFlow installation includes a tool, optimize_for_inference, that removes all nodes that aren't needed for a given set of input and outputs.

The script also does a few other optimizations that help speed up the model, such as merging explicit batch normalization operations into the convolutional weights to reduce the number of calculations. This can give a 30% speed up, depending on the input model. Here's how you run the script:

> python -m tensorflow.python.tools.optimize_for_inference \
>  --input=tf_files/retrained_graph.pb \
>  --output=tf_files/optimized_graph.pb \
>  --input_names="Mul" \
>  --output_names="final_result"

Quantize the network weights: Applying an almost identical process to your neural network weights has a similar effect. It gives a lot more repetition for the compression algorithm to take advantage of, while reducing the precision by a small amount (typically less than a 1% drop in precision).

It does this without any changes to the structure of the network, it simply quantizes the constants in place.

Now use the quantize_graph script to apply these changes:

> python -m scripts.quantize_graph \
>  --input=tf_files/optimized_graph.pb \
>  --output=tf_files/rounded_graph.pb \
>  --output_node_names=final_result \
>  --mode=weights_rounded

## Add your model files to the project
The demo project is configured to search for a graph.pb, and a labels.txt files in the android/tfmobile/assets directory. Replace those two files with your versions. The following command accomplishes this task:

> cp tf_files/rounded_graph.pb android/tfmobile/assets/graph.pb

> cp tf_files/retrained_labels.txt android/tfmobile/assets/labels.txt 

Structure demo i share in this repo, you can download and test it.

## Demo

## Full source code
All of step you need at here: https://bit.ly/2ILVkfv

## Reference:
* Tensorflow-for-poets-2: https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0
* Tensorflow mobile: https://www.tensorflow.org/mobile/ & https://www.tensorflow.org/tutorials/image_recognition 
* Pete Warden's blog: https://petewarden.com/2016/09/27/tensorflow-for-mobile-poets/

## Thanks for your reading. It help? Vote star? :star: 
See another repo and get issues if you get any question.

Made by Damminhtien from HUST :heart:

Happy training! :octocat:
