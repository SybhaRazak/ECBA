import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def predict_image(model, image_path, class_names):
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    class_probability = predictions[0][predicted_class]

    predicted_class_name = class_names[predicted_class]

    return predicted_class_name, class_probability, img

# Example usage (assuming you have your model and class names)
# Assuming 'model' is your trained model and 'class_names' is a list of your class labels
uploaded_image_path = "/content/speed-limit-sign-number-fifty-260nw-224034151.webp"  # Example path

predicted_class, probability, img = predict_image(model, uploaded_image_path, class_names)

# Display the image
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()

# Print prediction results
print(f"Predicted class: {predicted_class}")
print(f"Probability: {probability}")
