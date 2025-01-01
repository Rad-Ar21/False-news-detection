import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from requests import get
response = get("https://storage.yandexcloud.net/academy.ai/practica/fake_news.csv")
with open('fake_news.csv', 'wb') as f:
    f.write(response.content)

df = pd.read_csv(r"/content/fake_news.csv")

X_train,X_test,y_train,y_test=train_test_split(df['text'], df.label, test_size=0.2, random_state=7)

tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

tfidf_train=tfidf_vectorizer.fit_transform(X_train)
tfidf_test=tfidf_vectorizer.transform(X_test)

pac=PassiveAggressiveClassifier()
pac.fit(tfidf_train,y_train)

y_pred=pac.predict(tfidf_test)

score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=7)
ValScore=cross_val_score(pac,tfidf_train, y_train, cv=cv)
for i in range (0,5):
    print(f'Cross-validation score, fold {i+1}: {round(ValScore[i]*100,2)}%')
#Визуализация матрицы ошибок
fig, ax = plt.subplots(figsize=(10, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, colorbar=True)
plt.show()
