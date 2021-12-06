import tensorflow as tf
import numpy as np 
from tensorflow import keras
import os
from typing import Dict, List, Any
import pickle
from PIL import Image
class PreTrainedPipeline():
    def __init__(self, path=""):
        
        # load the model
        self.decoder = keras.models.load_model(os.path.join(path, "decoder"))
        self.decoder = keras.models.load_model(os.path.join(path, "encoder"))
        
        image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
        new_input = image_model.input
        hidden_layer = image_model.layers[-1].output

        self.image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
        
        with open('./tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)



    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.io.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path

    def __call__(self, inputs: "Image.Image") -> List[Dict[str, Any]]:
        """
        Args:
            inputs (:obj:`PIL.Image`):
                The raw image representation as PIL.
                No transformation made whatsoever from the input. Make all necessary transformations here.
        Return:
            A :obj:`list`:. The list contains items that are dicts should be liked {"label": "XXX", "score": 0.82}
                It is preferred if the returned list is in decreasing `score` order
        """

        hidden = tf.zeros((1, 512))

        temp_input = tf.expand_dims(load_image(image)[0], 0)
        img_tensor_val = self.image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                    -1,
                                                    img_tensor_val.shape[3]))

        features = self.encoder(img_tensor_val)

        dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
        result = []

        for i in range(max_length):
            predictions, hidden, attention_weights = self.decoder(dec_input,
                                                            features,
                                                            hidden)

            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(self.tokenizer.index_word[predicted_id])

            if self.tokenizer.index_word[predicted_id] == '<end>':
                return result

            dec_input = tf.expand_dims([predicted_id], 0)
        return result
