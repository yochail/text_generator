from gensim.models.fasttext import FastText
from gensim.models.word2vec import LineSentence


def train_model(corpus_path: str, output_path: str):
	sentences = LineSentence(corpus_path, limit=None)
	model = FastText(size=100, min_n=3, max_n=5, window=5, min_count=0)
	model.build_vocab(sentences=sentences)
	model.train(sentences=sentences, total_words=10000, epochs=10)
	model.save(output_path)


if __name__ == "__main__":
	train_model("Data/tanah/all.txt", "data/models/tanah_ft")
