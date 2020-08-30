from sentiment import Sentiment
import pandas as pd
import numpy as np

obj = Sentiment()
obj.movie_reviews = pd.read_csv('test.csv')
obj.glove_file = open('../../../IMDB Dataset/glove.6B.50d.txt', encoding="utf8")
obj.prepare_data()
obj.create_model()
print(obj.model.predict(obj.X_train))
print(obj.model.predict(obj.X_test))