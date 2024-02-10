### Add lines to import modules as needed
import tensorflow as tf
from tensorflow.keras import layers, models, datasets, utils, Input, Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Input, add, Dropout, SeparableConv2D, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
## 

# Open the image file
img = tf.io.read_file('/home/csalitre/school/machine-learning/ECGR-4127/simple-convolutional-neural-networks-Salitre97/dog.jpg')
img = tf.image.decode_image(img, channels=3)

# Resize the image
img_resized = tf.image.resize(img, [32, 32])

# Save the resized image 
img_resized = tf.cast(img_resized, tf.uint8).numpy()

img_pil = Image.fromarray(img_resized)
img_pil.save('/home/csalitre/school/machine-learning/ECGR-4127/simple-convolutional-neural-networks-Salitre97/dog_resized.jpg')

def plot_loss(history, title="Model Loss", save_plot=False, file_name="loss_plot.png"):
   plt.figure(figsize=(8,6)) 
   plt.plot(history.history['loss'], label='Training Loss')
   plt.title(history.history['val_loss'], label='Validation Loss')
   plt.title(title)
   plt.ylabel('Loss')
   plt.xlabel('Epoch')
   plt.legend()
   if save_plot:
      plt.save_plot(file_name)
   else: 
      plt.show()

def build_model1():
  model = tf.keras.Sequential([
    Input(shape=(32,32,3)),
    layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
    layers.BatchNormalization(),
    
    layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
    layers.BatchNormalization(),
    
    layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu'),
    layers.BatchNormalization(),
    
    layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),  
    layers.BatchNormalization(),

    layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),  
    layers.BatchNormalization(),

    layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),  
    layers.BatchNormalization(),

    layers.Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'),  
    layers.BatchNormalization(),
    layers.MaxPooling2D((4,4), strides=(4,4)), 
    layers.Flatten(),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(10)
  ])
  return model

def build_model2():
  model = tf.keras.Sequential([
    Input(shape=(32,32,3)),
        Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu'),
        BatchNormalization(),

        SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        BatchNormalization(),
        SeparableConv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu'),
        BatchNormalization(),
        SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        SeparableConv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
  ])
  return model

def build_model3():
    inputs = Input(shape=(32, 32, 3))
    x = Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    
    # Adjusted filter sizes for parameter reduction
    filter_sizes = [64, 128, 128, 128, 128, 128]  # Example adjustment
    previous_block_output = x
    add_counter = 0

    for filters in filter_sizes:
        x = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)  # Regularization

        # Implementing skip connection logic
        if previous_block_output.shape[-1] != x.shape[-1]:  # Condition for adding skip connection
            residual = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', use_bias=False)(previous_block_output)
            x = add([x, residual])
            add_counter += 1  # Keeping track for debugging purposes
        previous_block_output = x

    x = GlobalAveragePooling2D()(x)  # Reduced parameters before Dense layer
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(10, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(f"Add operations: {add_counter}")  # Debugging line to verify the number of Add operations
    return model

def build_model50k():
  model = tf.keras.Sequential([
        # Input layer
        Input(shape=(32, 32, 3)),
        
        # First depthwise-separable convolution
        Conv2D(64, (3, 3), padding='same', strides=(2,2), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Second depthwise-separable convolution
        SeparableConv2D(64, (3, 3), padding='same', strides=(2,2), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Third depthwise-separable convolution
        SeparableConv2D(128, (3, 3), padding='same', strides=(2,2), activation='relu'),
        BatchNormalization(),
        
        # Global Average Pooling instead of Flatten to reduce parameters
        layers.GlobalAveragePooling2D(),
        
        # Dense layer
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        # Output layer
        Dense(10, activation='softmax')
    ])
  return model

# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set

  # Load the CIFAR-10 dataset
  (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

  # Normalize the images
  train_images, test_images = train_images / 255.0, test_images / 255.0

  # Split the training data into training and validation subsets
  train_images, val_images, train_labels, val_labels = train_test_split(
      train_images, train_labels, test_size=0.2, random_state=42
  )
  '''
  ########################################
  ## Build and train model 1
  model1 = build_model1()

  # compile and train model 1.
  model1.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
  model1.summary()

  train_history_model1 = model1.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

  #plot_loss(train_history_model1, title='Model 1 Loss', save_plot=True, file_name="model1_loss.png")

  # Evaluate on the test set
  test_loss, test_accuracy = model1.evaluate(test_images, test_labels)
  print(f"Test Accuracy: {test_accuracy}")

  # Classify Dog image

  # Load image
  img = load_img('/home/csalitre/school/machine-learning/ECGR-4127/simple-convolutional-neural-networks-Salitre97/dog_resized.jpg')
  img_array = img_to_array(img) /255.0
  img_array_expanded = np.expand_dims(img_array, axis=0)
  predictions = model1.predict(img_array_expanded)
  predicted_class = np.argmax(predictions, axis=1)

  # CIFAR=10 class labels
  class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

  # Output the predicted class
  print("Predicted class:", class_labels[predicted_class[0]])


 #########################################################
  ## Build, compile, and train model 2 (DS Convolutions)
  model2 = build_model2()

  # compile and train model 1.
  model2.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
  model2.summary()

  train_history_model2 = model2.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

  #plot_loss(train_history_model2, title='Model 2 Loss', save_plot=True, file_name="model2_loss.png")

  # Evaluate on the test set
  test_loss_model2, test_accuracy_model2 = model2.evaluate(test_images, test_labels)
  print(f"Test Accuracy: {test_accuracy_model2}")
  '''
  ###########################################################
  ### Repeat for model 3 and your best sub-50k params model
  model3 = build_model3()

  # compile and train model 1.
  model3.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
  model3.summary()

  train_history_model3 = model3.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))

  #plot_loss(train_history_model3, title='Model 1 Loss', save_plot=True, file_name="model3_loss.png")

  # Evaluate on the test set
  test_loss_model3, test_accuracy_model3 = model3.evaluate(test_images, test_labels)
  print(f"Test Accuracy: {test_accuracy_model3}")

  '''
  #############################################################
  model50k = build_model50k()

  # compile and train model 1.
  model50k.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
  model50k.summary()

  train_history_model50k = model50k.fit(train_images, train_labels, epochs=50, validation_data=(val_images, val_labels))
  
  #plot_loss(train_history_model50k, title='Model 2 Loss', save_plot=True, file_name="model50k_loss.png")

  # Evaluate on the test set
  test_loss_model50k, test_accuracy_model50k = model50k.evaluate(test_images, test_labels)
  print(f"Test Accuracy of model 50k: {test_accuracy_model50k}")
  
  model50k.save('best_model.h5')
  '''