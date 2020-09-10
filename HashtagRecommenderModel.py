import numpy as np
import pandas as pd
import os

from scipy.spatial.distance import cosine
import pickle
from scipy.sparse import csr_matrix

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

def split_file(source, dest_folder, write_size):
    # Make a destination folder if it doesn't exist yet
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)
    else:
        # Otherwise clean out all files in the destination folder
        for file in os.listdir(dest_folder):
            os.remove(os.path.join(dest_folder, file))
    partnum = 0
    
    # Open the source file in binary mode
    input_file = open(source, 'rb')
    while True:
        # Read a portion of the input file
        chunk = input_file.read(write_size)
        
        # End the loop if we have hit EOF
        if not chunk:
            break
        
        # Increment partnum
        partnum += 1
        
        # Create a new file name
        filename = os.path.join(dest_folder, f'final_model_{partnum}.pkl_part')
        
        # Create a destination file
        dest_file = open(filename, 'wb')
        
        # Write to this portion of the destination file
        dest_file.write(chunk)
        # Explicitly close 
        dest_file.close()
    # Explicitly close
    input_file.close()
    # Return the number of files created by the split
    return partnum


def join_file(source_dir, dest_file):
    # Create a new destination file
    output_file = open(dest_file, 'wb')    
 
    # Go through each portion one by one
    
    for f in sorted(os.listdir(source_dir)):
         
        # Assemble the full path to the file
        path = os.path.join(source_dir, f)
         
        # Open the part
        input_file = open(path, 'rb')
         
        while True:
            # Read all bytes of the part
            bytes = input_file.read()
             
            # Break out of loop if we are at end of file
            if not bytes:
                break
                 
            # Write the bytes to the output file
            output_file.write(bytes)
             
        # Close the input file
        input_file.close()
         
    # Close the output file
    output_file.close()

def get_top_hashtags(hashtags_scores, num_predict, all_hashtags):
    top_hashtags = [all_hashtags[hashtag_id] for hashtag_id in hashtags_scores.argsort()[::-1][:num_predict]]
    return top_hashtags


def preprocess_image(image_path, image_shape):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image)
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (image_shape[0], image_shape[1]))
    # Reshape grayscale images to match dimensions of color images
    if image.shape != image_shape:
        image = tf.concat([image, image, image], axis=2)
    image = tf.reshape(image, (1, *image_shape)).numpy()
    return image


def extract_features(images_paths):
    IMAGE_SHAPE = (160, 160, 3)
    # Create the base model from the pre-trained model MobileNet V2
    base_model = MobileNetV2(input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    net = tf.keras.Sequential([
      base_model,
      global_average_layer,
    ])

    images = np.concatenate([preprocess_image(image_path, IMAGE_SHAPE) for image_path in images_paths])
    images_features = list(net.predict(images))
    return images_features

class HashtagRecommender:
    
    def __init__(self, data_path=''):
        
        # paths for the data (images, metadata and images features)
        self.data_path = data_path
        self.base_images_path = os.path.join(data_path, 'images', '')
        self.metadata_path = os.path.join(data_path, 'metadata.csv')
        self.full_data_path = os.path.join(data_path, 'full_data.pkl')
        
        self.metadata = self.load_metadata()

        # generate the images names and lookup
        self.generate_images_names()
        
        # load full data
        self.load_full_data()

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__ = d
            
    def generate_hashtags_indicators(self):
        hashtags_indicators = np.zeros((self.data_size, self.num_hashtags), dtype=np.int32)
        for i, image_name in enumerate(self.images_names):
            for hashtag in self.full_data.hashtags.loc[image_name]:
                hashtags_indicators[i, self.all_hashtags_lookup[hashtag]] += 1
        hashtags_indicators = list(hashtags_indicators)
        self.full_data['hashtags_indicators'] = hashtags_indicators
    
    def load_metadata(self):
        metadata = pd.read_csv(self.metadata_path)
        metadata['hashtags'] = metadata.hashtags.apply(eval)
        metadata.set_index('image_name', inplace=True)
        return metadata

    def load_images_features(self):
        images_paths = [self.base_images_path + image_name for image_name in self.images_names]
        images_features = extract_features(images_paths)
        return images_features
    
    def load_full_data(self):
        if os.path.isfile(self.full_data_path):
            self.full_data = pd.read_pickle(self.full_data_path)
            self.generate_all_hashtags()
            self.data_size = self.full_data.shape[0]
        else:
            self.full_data = self.metadata.copy()
            self.data_size = self.full_data.shape[0]
            self.generate_all_hashtags()
            self.generate_hashtags_indicators()
            image_features = self.load_images_features()
            self.full_data['images_features'] = image_features
            self.full_data.to_pickle(self.full_data_path)
            
    
    def generate_all_hashtags(self):
        self.all_hashtags = sorted({hashtag for hashtags_list in  self.full_data.hashtags.values for hashtag in hashtags_list})
        self.all_hashtags_lookup = {hashtag: i for i, hashtag in enumerate(self.all_hashtags)}
        self.num_hashtags = len(self.all_hashtags)
    
    def generate_images_names(self):
        self.images_names = list(set(os.listdir(self.base_images_path)).intersection(self.metadata.index))
        self.images_names_lookup = {image_name: i for i, image_name in enumerate(self.images_names)}
    
    def get_als_data(self):
        als_data = []
        for image_name in self.train_images_names:
            hashtag_list = self.train_data.loc[image_name, 'hashtags']
            for hashtag in hashtag_list:
                als_data.append({'image_id': self.images_names_lookup[image_name], 'hashtag_id': self.all_hashtags_lookup[hashtag], 'rating': 1})
        als_data = pd.DataFrame(als_data)
        return als_data
    
    def set_train(self, images_names):
        self.train_images_names = images_names
        self.train_data = self.full_data.loc[images_names]
        self.generate_train_hashtags_similarity()
    
    def fit(self, images_names):
        from pyspark.sql import SparkSession
        from pyspark.ml.recommendation import ALS
        self.set_train(images_names)
        spark = SparkSession.builder.master('local').getOrCreate()
        als_data = spark.createDataFrame(self.get_als_data())
        als = ALS(userCol='image_id', itemCol='hashtag_id', implicitPrefs=True)
        als.setSeed(0)
        als_model = als.fit(als_data)
        self.als_hashtags_features = als_model.itemFactors.toPandas().set_index('id')['features'].sort_index()
        self.als_images_features = als_model.userFactors.toPandas().set_index('id')['features'].sort_index()
        self.als_images_features.index = self.als_images_features.index.map(lambda x: self.images_names[x])
       
    def get_hashtags_scores(self, image_features, selected_hashtags, alpha, num_neighbors):
        image_features = np.expand_dims(image_features, axis=0)
        image_similarity = self.train_data.images_features.apply(lambda x: cosine(x, image_features))
        similar_images = list(image_similarity.sort_values()[:num_neighbors].index)
        mean_similar_images_als_features = np.vstack(self.als_images_features[similar_images]).mean(axis=0)
        hashtags_scores = self.als_hashtags_features.apply(lambda x: np.dot(x, mean_similar_images_als_features)).sort_index().values
        
        consider_input_hashtag = (selected_hashtags is not None 
                and all(hashtag in self.all_hashtags_lookup for hashtag in selected_hashtags))
        if consider_input_hashtag:
            selected_hashtags_indices = [self.all_hashtags_lookup[selected_hashtag] for selected_hashtag in selected_hashtags]
            hashtags_scores[selected_hashtags_indices] = -np.inf
        
        hashtags_scores = np.exp(hashtags_scores)
        hashtags_scores /= hashtags_scores.sum()
        
        if consider_input_hashtag:
            for selected_hashtags_index in selected_hashtags_indices:
                hashtags_scores = alpha * hashtags_scores + (1 - alpha) * self.train_hashtags_similarity[selected_hashtags_index]
            for selected_hashtags_index in selected_hashtags_indices:
                hashtags_scores[selected_hashtags_index] = 0
        return hashtags_scores
    
    def predict_scores(self, images_names, selected_hashtags, alpha, num_neighbors=20):
        if type(images_names) is list:
            images_features = self.full_data.images_features[images_names]
        else:
            images_names = [images_names]
            images_features = np.array(extract_features(images_names))
        hashtags_scores = [self.get_hashtags_scores(image_features, selected_hashtags, alpha, num_neighbors) for image_features in images_features]
        hashtags_scores = pd.Series(hashtags_scores, index=images_names)
        return hashtags_scores
    
    def predict_hashtags(self, images_names, num_neighbors=20, num_predict=5, selected_hashtags=None, alpha=0.5):
        hashtags_scores = self.predict_scores(images_names, selected_hashtags, alpha, num_neighbors)
        
        hashtags_predictions = hashtags_scores.apply(lambda scores: get_top_hashtags(scores, num_predict, self.all_hashtags))
        return hashtags_predictions
    
    
    def generate_train_hashtags_similarity(self):
        train_hashtags_indicators = np.stack(self.train_data.hashtags_indicators.values)
        sparse_train_hashtags_indicators = csr_matrix(train_hashtags_indicators)
        self.train_hashtags_similarity = (sparse_train_hashtags_indicators.T @ sparse_train_hashtags_indicators).toarray()
        np.fill_diagonal(self.train_hashtags_similarity, 0)
        self.train_hashtags_similarity = self.train_hashtags_similarity / np.log(train_hashtags_indicators.sum(axis=0)[None, :])
        self.train_hashtags_similarity = self.train_hashtags_similarity / self.train_hashtags_similarity.sum(axis=1)[:, None]
        self.train_hashtags_similarity = np.nan_to_num(self.train_hashtags_similarity)

    def evaluate(self, hashtags_predictions):
        mrr = 0
        hashtags_true = self.full_data.hashtags.loc[hashtags_predictions.index]
        for i, image_name in enumerate(hashtags_predictions.index):
            rr = 1 / (min([((hashtag_prediction not in hashtags_true.loc[image_name]), i) for i, hashtag_prediction in enumerate(hashtags_predictions.loc[image_name])])[1] + 1)
            mrr += rr
        mrr /= hashtags_predictions.shape[0]
        return mrr


if __name__ == '__main__':
    data_path = '../data'
    hr = HashtagRecommender(data_path=data_path)
    hr.fit(hr.images_names)
    pickle.dump(hr, open('model.pkl', 'wb'))
    split_file(source='model.pkl', write_size=50*(10**6), dest_folder='Model_Files')
