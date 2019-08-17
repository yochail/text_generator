import numpy as np
import tensorflow as tf
from gensim.models import FastText
from gensim.models.keyedvectors import FastTextKeyedVectors
from gensim.models.word2vec import LineSentence
from tensorflow.python import Constant
from tensorflow.python.keras.layers import LSTM, Activation, Dense, Dropout, Embedding, Bidirectional
from tensorflow.python.keras.losses import cosine_proximity


def create_gen_model(embeding, Xs):
	pretrained_weights = embeding.wv.syn0
	vocab_size, emdedding_size = pretrained_weights.shape
	model = tf.keras.Sequential([
		Embedding(input_dim=vocab_size,
		          output_dim=emdedding_size,
		          weights=[pretrained_weights],
		          embeddings_initializer=Constant(embeding.wv.syn0),
		          mask_zero=True),
		Bidirectional(LSTM(units=emdedding_size, return_sequences=False)),
		# ,input_shape=Xs.shape[1:], return_sequences=False),
		# Dropout(0.2),
		# LSTM(units=vec_length, return_sequences=False),
		Dropout(0.4),
		Dense(units=vocab_size, activation='relu'),
		# Dropout(rate=0.5),
		# model.add(TimeDistributed(Dense(1)))
		Activation('softmax')
	])
	return model


def create_training_data(data: LineSentence,
                         seq_length: int,
                         word2vec) \
		-> (np.array, np.array):
	lastWords = np.zeros(seq_length
	                     # , word2vec.vector_size)
	                     ).tolist()
	Xs = []
	ys = []
	for sen in data:
		for word in sen.replace('\n', '').split(' '):
			y = word2vec.vocab[word].index
			# y = word2vec[word].tolist()
			Xs.append(lastWords)
			ys.append(y)
			lastWords = lastWords[1:]
			lastWords.append(y)
	return np.array(Xs), np.array(ys)


def get_closest_word_idx(embeding: FastTextKeyedVectors, word):
	if (word in embeding.wv.vocab):
		return embeding.wv.vocab[word].index
	else:
		closest = embeding.wv.similar_by_word(word, 1)
		return embeding.wv.vocab[closest[0][0]].index


def print_test_prediction(model, embeding, text: str, length: int):
	text_arr = np.array(text.split(' '))
	test_pred = np.array(
		[get_closest_word_idx(embeding, x) for x in text_arr])
	# [np.zeros((5,100)) for x in [text_arr]])

	finel = text
	pred_word = ""
	for i in range(length):
		predictionVec = model.predict(test_pred)
		predictionIdx = sample(predictionVec[-1], temperature=0.3)
		new_pred_word = embeding.wv.index2word[predictionIdx]
		# new_pred_word = embeding.wv.index2word(prediction[0],2)
		# print(predictionIdx)
		# pred_word = [w for w in new_pred_word if w[0] != pred_word][0]
		# print(#f"{text_arr[0]}"
		# f" {text_arr[1]}"
		#     f" {pred_word}")
		finel += " " + new_pred_word
		# text_arr = np.delete(text_arr, 0, 0)
		# text_arr = np.append(text_arr, [pred_word], 0)
		# test_pred = np.delete(test_pred, 0, 0)
		test_pred = np.append(test_pred, [predictionIdx], 0)
	print(finel)


def sample(preds, temperature=1.0):
	if temperature <= 0:
		return np.argmax(preds)
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)


def cosine_distanse(y_true, y_pred):
	val = 1 - cosine_proximity(y_true, y_pred)
	print(val)
	return val


if __name__ == "__main__":
	sentence_lines = open(text_corpus_path, encoding="utf-8")
	embeding = FastText.load(fastext_vectors_path)

	batch_size = 1
	epochs = 10
	window = 20
	generate_n = 50
	Xs, ys = create_training_data(sentence_lines, window, embeding.wv)
	model = create_gen_model(embeding, Xs)
	model.compile(optimizer='adam',
	              loss="sparse_categorical_crossentropy",  # cosine_distanse,
	              metrics=['accuracy'])
	model.fit(Xs, ys, epochs=epochs, shuffle=False)
	test_text = "בראשית ברא אלהים את השמים".join(" ".split(' ')[:window - 1])
	print_test_prediction(model, embeding, test_text, generate_n)

	test_text = "ונח היה איש צדיק תמים".join(" ".split(' ')[:window - 1])
	print_test_prediction(model, embeding, test_text, generate_n)

	test_text = "לך לך מארצך וממולדתך".join(" ".split(' ')[:window - 1])
	print_test_prediction(model, embeding, test_text, generate_n)

	test_text = "ותלך כיפה אדומה ביער הזאבים".join(" ".split(' ')[:window - 1])
	print_test_prediction(model, embeding, test_text, generate_n)
