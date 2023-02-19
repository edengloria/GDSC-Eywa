import tensorflow as tf
from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import numpy as np

# Load the MobileNet model
model = load_model('Model\mobilenet.h5')

# Load and preprocess the input image
img_path = 'path/to/image.jpg'
img = load_img(img_path, target_size=(224, 224))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Perform inference on the image
preds = model.predict(x)

# Decode the predictions into class labels
class_labels = decode_predictions(preds, top=1)[0]
class_id, class_name, class_score = class_labels[0]

# Print and store the predicted class label
print(f'Predicted class: {class_name} (score: {class_score:.2f})')
with open('predicted_class.txt', 'w') as f:
    f.write(class_name)
