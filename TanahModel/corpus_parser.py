import json
import os


def json_to_line_sentence(sourcedir: str, desfile: str):
	files = os.listdir(sourcedir)
	output = open(desfile, "w", encoding="utf-8")
	for idx, file_name in enumerate(files):
		print(f"Reading json file: {sourcedir}/{file_name}")
		with open(f"{sourcedir}/{file_name}", encoding="utf-8") as json_file:
			lines = []
			json_data = json.load(json_file)
			for peregraph in json_data["text"]:
				for sentence in peregraph:
					lines.append(sentence.replace("־", " ") + "\n")

			output.writelines(lines, )


if __name__ == "__main__":
	json_to_line_sentence("Data/tanah/files", "Data/tanah/all.txt")
