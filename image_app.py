import streamlit as st
import numpy as np
import pandas as pd
import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity

# path
path = os.path.dirname(__file__)

# load feature vectors
data = np.load('all_vecs.npz')
all_vecs = data['a']
all_vecs = all_vecs.reshape(len(all_vecs), -1)

# load furniture dataset with links
furns = pd.read_csv(path + '/furns.csv')

# load full model and model for similarity search
model = load_model(path + "/furn.h5", compile=False)

# loaded_model = load_model(path + "/furn.h5",compile=False)
layers_to_load = model.layers[:-3]
new_model = tf.keras.models.Sequential(layers_to_load)
new_model.build((None,128,128,3))
new_model.load_weights(path + '/furn_weights.h5')

# Function to preprocess the uploaded image
def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = image.resize((128, 128))  # Resize the image to match the input size of the model
    image = np.array(image)  # Convert the image to a NumPy array
    image = np.expand_dims(image, axis=0)  # Add an extra dimension to represent the batch
    return image
    
# Function to classify the image
def classify(x):
    d = {}
    pred = model.predict(x)
    if pred[0][0] > pred[0][1]:
        d['small_table'] = 1
        d['sofa'] = 0
    else:
        d['small_table'] = 0
        d['sofa'] = 1
    return d
    
# Function to perform image search
def perform_image_search(query_image):
    query_embedding = new_model.predict(query_image)  # Get the embedding of the query image

    # Compute cosine similarity between query embedding and all embeddings
    similarities = cosine_similarity(query_embedding.reshape(1,-1),all_vecs)

    # Get the indices of the top similar images
    top_indices = np.argsort(similarities.flatten())[::-1][:5]

    # Get the URLs of the top similar images
    top_urls = furns.iloc[top_indices]['product_image'].tolist()

    return top_urls

# Streamlit app
def main():
    st.title("Image Search Engine")
    st.subheader("Input --> Images of small tables or sofas")
    # Upload image
    image = st.file_uploader("Upload an image", type=['jpeg'])

    if image is not None:
        st.image(image, use_column_width=True, width=300)

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Classify it
        result = classify(processed_image)  
        max_key = max(result, key=result.get)
        
        # Perform image search
        top_images = perform_image_search(processed_image)

        # Display the top similar images and their labels
        st.subheader(f"Image Classification: {max_key}")
        st.subheader(f"Top Similar Images")
        cols = st.columns(len(top_images))
        # for i in range(len(top_images)):
        for i, col in enumerate(cols):
            response = requests.get(top_images[i])
            image = Image.open(BytesIO(response.content))
            col.image(image, use_column_width=True)
            

if __name__ == '__main__':
    main()
