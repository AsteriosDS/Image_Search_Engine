## Image Search Engine
### This was built with the intent to replicate similar image search or a search by image functionality often found in various websites.
### 1. Approximately 50k living room furniture images scraped from the greek e-commerce website Skroutz. I ended up using close to 12k images belonging to two classes: small tables and sofas.
### 2. ~9k for training, the remaining split for testing and validation.
### 3. Built a CNN (tensorflow,optuna) for the classification task and used the same CNN minus the last 3 layers for feature vector extraction. Finally, vector similarity is calculated with cosine similarity.

### Repo contents:
#### 1. *Dataset_Builder_Furniture.ipynb*: Code used to scrape Skroutz and build the dataset.
#### 2. *Image_Search_Engine.ipynb*: Model building and refining.
#### 3. *all_vecs.npz*: Compressed numpy array with the extracted feature vectors from the training set (~9k).
#### 4. *furn.h5,furn_weights.h5*: CNN and its weights.
#### 5. *image_app.py*: Python code to build the streamlit app.
#### 6. *requirements.txt*: Requirements for the streamlit vm.
