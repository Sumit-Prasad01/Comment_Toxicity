import requests, zipfile, io

url = "http://nlp.stanford.edu/data/glove.6B.zip"
print("Downloading GloVe embeddings...")

r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("glove")  


print("Done! Now use glove/glove.6B.100d.txt")
