import numpy as np
import tensorflow as tf
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from tensorflow.python.keras.layers import LSTM, Activation

from TanahModel.config import *


def create_gen_model(vec_length, Xs):
	model = tf.keras.Sequential([
		LSTM(units=vec_length, input_shape=Xs.shape[1:], return_sequences=False),
		# Dropout(rate=0.5),
	# model.add(TimeDistributed(Dense(1)))
		Activation('softmax')])
	return model


def create_training_data(data: LineSentence,
                         seq_length: int,
                         word2vec) \
		-> (np.array, np.array):
	lastWords = np.zeros((seq_length, word2vec.vector_size)).tolist()
	Xs = []
	ys = []
	for sen in data:
		for word in sen.replace('\n', '').split(' '):
			y = word2vec[word].tolist()
			Xs.append(lastWords)
			ys.append(y)
			lastWords = lastWords[1:]
			lastWords.append(y)
	return np.array(Xs), np.array(ys)


def print_test_prediction(model, embeding, text: str, length: int):
	test_pred = np.array(
		[embeding.wv[x] for x in [text.split(' ')]])
	print(text)
	for i in range(length):
		prediction = model.predict(test_pred)
		pred_word = embeding.most_similar(prediction)[0]
		print(pred_word[0])
		test_pred = np.delete(test_pred, 0, 1)
		test = pred_word[0]
		test2 = embeding.wv[test]
		test_pred = np.append(test_pred, np.array([[embeding.wv[pred_word[0]].tolist()]]), 1)

if __name__ == "__main__":
	sentence_lines = open(text_corpus_path, encoding="utf-8")
	embeding = FastText.load(fastext_vectors_path)
	Xs, ys = create_training_data(sentence_lines, 5, embeding.wv)
	model = create_gen_model(embeding.vector_size, Xs)
	model.compile(optimizer='adam',
	              loss='mean_squared_error',
	              metrics=['accuracy'])

	test_text = "בראשית ברא אלוהים את השמים"
	print_test_prediction(model, embeding, test_text, 50)
