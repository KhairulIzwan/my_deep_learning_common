# USAGE
# python visualize_architecture.py

# import the necessary packages
from pyimagesearch.nn.conv.lenet import LeNet
from keras.utils import plot_model

# initialize LeNet and then write the network architecture
# visualization grpah to disk
model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file="lenet_architecture.eps", show_shapes=True)
