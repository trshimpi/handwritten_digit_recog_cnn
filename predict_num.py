import numpy as np
import base64
import io
from PIL import Image
from scipy.misc import imsave, imread, imresize
import tensorflow as tf
import keras
from keras.models import Sequential,load_model
from keras.layers import Flatten,Conv2D,Dense
from keras.preprocessing.image import ImageDataGenerator,img_to_array
import numpy as np

from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)

def get_model():
	global model
	model = load_model("mnist_keras_216_aug.h5")
	print(model.summary())
	# save the graph after loading the model
	global graph
	graph = tf.get_default_graph()
	print(" *model loaded")

def preprocessing(image,target_size):
	image = np.resize(image,target_size)
	image = image.reshape(1,28,28,1)
	# image = img_to_array(image)
	# image = np.expand_dims(image,axis=0)

	return image

print("* loading model")
get_model()

@app.route("/predict",methods=["POST"])
def predict():
	message = request.get_json(force=True)
	encoded = message["image"]
	decoded = base64.b64decode(encoded+ "========")
	filename = 'requested.png'  # I assume you have a way of picking unique filenames
	with open(filename, 'wb') as f:
		f.write(decoded)

	x = imread('requested.png',mode='L')
	#compute a bit-wise inversion so black becomes white and vice versa
	x = np.invert(x)
	#make it the right size
	x = imresize(x,(28,28))
	#imshow(x)
	#convert to a 4D tensor to feed into our model
	x = x.reshape(1,28,28,1)
	# processed_image = preprocessing(image,target_size=(224,224))

	with graph.as_default():
		prediction = model.predict(x).tolist()
		print(prediction)
		# print("shape==============================================")
		# print(prediction.shape)
		print("type===============================================")
		print(type(prediction))
		print("max================================================")
		print(prediction[0].index(max(prediction[0])))
		# print(np.argmax(prediction,axis=1)[0])

	response = {
	'prediction':{
					'val':prediction[0].index(max(prediction[0]))
				}
	}
	return jsonify(response)
