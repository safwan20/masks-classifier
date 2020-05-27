# masks-classifier

# About dataset  
The dataset has been taken from this repo https://github.com/prajnasb/observations.

# About model architecture  
layers : Conv2D--->MaxPool2D---->Conv2D--->MaxPool2D---->Conv2D--->MaxPool2D---->Flatten--->Dense---->Dropout--->Dense  
activation : (relu)              (relu)                  (relu)                            (relu)                (sigmoid)  
fliter : 32,(3,3)  
pool_size = (2,2)  
input_shape = (100,100,3)  
output_shape = 128  

# Metrics  
loss='binary_crossentropy'  
optimizer='adam'  
metrics=['accuracy']  

# Save architecture  
I have used h5 format for saving.
