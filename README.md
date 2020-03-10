# Computer Vision Lab (up to 5 points).

## Basic implementations (1 points)

Check basic implementations on CIFAR10 in the Deep Learning Lab project [here](https://github.com/RParedesPalacios/DeepLearningLab/tree/master/Examples/Keras/CIFAR)

![Cifar10](images/cifar10.png)

**Goals:**

* Implement some basic convolutional networks
* Implement different data augmentation
* Implement VGG model A

---

## Advanced topologies (2 points)

* Residual Nets [example of Keras](src/cifar10_resnet.py)

* Residual Nets [example of Pytorch](src/resnet.py)

* Wide Resnet 

* Dense Nets


---

## Gender Recognition (1 point)

Images from "Labeled Faces in the Wild" dataset (LFW) in realistic scenarios, poses and gestures. Faces are automatically detected and cropped to 100x100 pixels RGB.


![Face example](images/face.png)


**Training** set: 10585 images

**Test** set: 2648 images 


**Python Notebook**: [here](notebook/gender.ipynb)

**Python code**: [here](src/gender.py)

**Goals:**
* Implement a model with >95% accuracy over test set
* Implement a model with >90% accuracy with less than 100K parameters
  
  get some inspiration from [Paper](https://pdfs.semanticscholar.org/d0eb/3fd1b1750242f3bb39ce9ac27fc8cc7c5af0.pdf)
    

---

## Car Model identification with bi-linear models (2 points)

Images of 20 different models of cars.

![Cars](images/cars.png)

**Training** set: 791 images

**Test** set: 784 images 

* Version 1. Two different CNNs:

  **Python code**: [here](src/cars1.py)

* Version 2. The same CNN (potentially a pre-trained model)

  **Python code**: [here](src/cars2.py)

**Goals:**
* Understand the above Keras implementations:
  * Name the layers
  * Built several models
  * Understand tensors sizes
  * Connect models with operations (outproduct)
  * Create an image generator that returns a list of tensors
  * Create a data flow with multiple inputs for the model
  * Understand the limitations of the proposed solution

* **Lab Project (2 points of labs mark)**
  * Load a pre-trained VGG16 
  * Connect this pre-trained model and form a bi-linear
  * Train freezing weights or not
  * ...
  
[Paper](https://pdfs.semanticscholar.org/3a30/7b7e2e742dd71b6d1ca7fde7454f9ebd2811.pdf)


## Image colorization (1 point)

![Cars](images/color.png)

Code extracted and adapted from [github](https://github.com/emilwallner/Coloring-greyscale-images-in-Keras)

**Goals:**

* Use a more simple version from: [alpha version](https://github.com/emilwallner/Coloring-greyscale-images-in-Keras/tree/master/floydhub/Alpha-version)

* Use the full version. Code adapted to download images for training and test:

	**Python Notebook**: [here](notebook/colorization.ipynb)
	
		
	**Python code**: [here](src/colorization.py)

* Understand the above Keras implementations:
	* How to load the inception net 
	* How to merge encoder and inception result
	* Use image functions to obtain lab space
	* Create an appropiate  data augmentation 


Need help? [Read](https://blog.floydhub.com/colorizing-b-w-photos-with-neural-networks/)

## Style transfer (1 point)

![Transfer](images/transfer.png)

Code extracted and adapted from [github](https://github.com/dsgiitr/Neural-Style-Transfer)

Content image

![UPV](images/upv.png)

Style image

![Wave](images/style.png)

Result image

![Result](images/resultstyle.png)

**Python Notebook**: [here](notebook/style.ipynb)

**Python code**: [here](src/style.py)


## Other project? 

You are welcome!





















