timport streamlit as st
import numpy as np
import pandas as pd
import os
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import tensorflow as tf
from keras.models import model_from_json
from keras.initializers import glorot_uniform
from sklearn.metrics.pairwise import cosine_similarity

# path
path = os.path.dirname(__file__)

# load all_embeddings, links
all_embeddings = np.load(path + '/all_embeddings.npy')
links = pd.read_csv(path + '/links.csv')

#Reading the encoder from JSON file
with open(path + '/encoder.json', 'r') as json_file:
    json_savedModel= json_file.read()
    
#load the model architecture 
encoder = tf.keras.models.model_from_json(json_savedModel)

# Upload an image file
url = st.file_uploader("Paste the link of a jpeg", type="jpeg")

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize the image to match the input size of the model
    image = np.array(image)  # Convert the image to a NumPy array
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add an extra dimension to represent the batch
    return image

# Function to perform image search
def perform_image_search(query_image):
    query_embedding = encoder.predict(query_image)  # Get the embedding of the query image

    # Compute cosine similarity between query embedding and all embeddings
    similarities = cosine_similarity(all_embeddings, query_embedding)

    # Get the indices of the top similar images
    top_indices = np.argsort(similarities.flatten())[::-1][:5]

    # Get the URLs of the top similar images
    top_urls = links.iloc[top_indices].values

    return top_urls

# Streamlit app
def main():
    st.title("Image Search Engine")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Perform image search
        top_images = perform_image_search(processed_image)

        # Display the top similar images and their labels
        st.subheader("Top Similar Images")
        for i in range(len(top_images)):
            st.image(top_images[i], use_column_width=True)

if __name__ == '__main__':
    main()
