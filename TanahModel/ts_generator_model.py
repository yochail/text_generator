from typing import Callable

import numpy as np
import tensorflow as tf
from gensim.models.word2vec import LineSentence
from tensorflow.python.keras.layers import LSTM, Dropout, Activation

from TanahModel.config import *


def create_gen_model(input_shape):
	model = tf.keras.Sequential()
	model.Add(LSTM(8, input_shape=input_shape, return_sequences=False))
	model.add(Dropout(0.5))
	# model.add(TimeDistributed(Dense(1)))
	model.add(Activation('softmax'))
	return model


def create_training_data(data: LineSentence,
                         seq_length: int,
                         word2vec: Callable[[str], np.array]) \
		-> (np.array[str], np.array[str]):
	Xs = np.array()
	ys = np.array()
	lastWords = np.array(word2vec("") * seq_length)
	for sen in data:
		for word in sen.split(' '):
			X = lastWords
			y = word2vec(word)
			Xs.append(lastWords)
			ys.append(y)
			lastWords = lastWords[1:] + y
	return Xs, ys


if __name__ == "__main__":
	word_vector_path = ""

	sentence_lines = open(text_corpus_path)
	Xs, ys = create_training_data(text_corpus_path, 5)
	model = create_gen_model(word_vector_path)
