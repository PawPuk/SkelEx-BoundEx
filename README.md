# SkelEx-BoundEx
This repository contains the code for the paper [SkelEx and BoundEx: Natural Visualization of ReLU Neural Networks](https://arxiv.org/abs/2305.05562) 
writen by Pawel Pukowski and Haiping Lu from the University of Sheffield. The link to the paper will be added soon. 
This work allows us to transfer the weights and biases encoding into an encoding using the skeleton formed by the 
critical points of the learned functions, allowing for natural visualization of the functions learned by ReLU NNs
trained on 2D data. 

## Issues

1. The algorithm itself generalizes to any number of dimensions, but in our implementation we use the Shapely library 
which only supports operations on 2D polytopes. Any help in generalizing the code is much appreciated.
2. Shapely suffers from floating point error, which means that the code might sometimes not work. If this happens please
retrain the network. If the issue keeps persisting, then the network is most likely too large. Any help with fixing
those issues is much appreciated.
3. The implementation is currently working only on binary classification problem due to outdated implementation. The
code does not use the algorithm for generating BoundEx from the paper, which makes it work only on 2 class problems.
Any help with fixing this issue is much appreciated.

## Set Up

In terminal run:
```
git clone https://github.com/PawPuk/SkelEx-BoundEx.git
cd SkelEx-BoundEx
pip install -r requirements.txt
```

## Working example

*main.py* is the most important program from which we run all the necessary code.

We set *number_of_classes* and *number_of_parameters* to **2** as it's the only working case for now. We create a ReLU 
NN with architecture **2-10-10-2**, train for **50** epochs, and set the size of data clusters as **50**. You can 
make the architecture smaller to reduce the risk of the floating point error propagating and causing a bug.
```
number_of_classes = 2
number_of_parameters = 2
layers_width = [number_of_parameters, 10, 10, number_of_classes]
data_size = 1000
number_of_epochs = 50
```
We also set *dataset* to **None** in order to be working on the spiral data.
```
dataset = None
```
When we generate the data we use 3:1 ratio for the train:test data
```
my_data = (Dataset2D(class_size=3*data_size).data.to(dev), Dataset2D(class_size=data_size).data.to(dev))
```
**Visualizer** class from *visualisation.py* contains all the code for visualizing the results. We first prepare the 
graph for 2D visualisation. Next we add the learned skeleton to the *ax*. After that we draw the decision boundary.
Next two lines generate 3D visualisation of the learned loss landscape (for class 1), and the membership landscape.
```
visualizer = Visualizer(skeletons_of_learned_decision_functions, trained_model, hyperrectangle, number_of_classes)
ax = visualizer.prepare_graph("Skeleton tessellation")  # set rotation=180 to get image from Figure 1
visualizer.plot_skeleton(None, ax)
visualizer.plot_decision_boundary(boundary_extractor, classification_polygons, lines_used, ax, save=True)
visualizer.draw_loss_landscape(25, 50, 0, class_index=1, save=True)   # set class_index=1 for blue class
visualizer.draw_decision_landscape(25, 50, 0, skeleton=False, decision=True, heatmap=False, save=True)
```

To learn more about the options available in the Visualizer class please read the documentation of Visualizer's methods.
