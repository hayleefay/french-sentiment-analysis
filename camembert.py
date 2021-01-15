from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine")
model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")

nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# sanity checks
print(nlp("Alad'2 est clairement le meilleur film de l'année 2018.")) # POSITIVE
print(nlp("Juste whoaaahouuu !")) # POSITIVE
print(nlp("NUL...A...CHIER ! FIN DE TRANSMISSION.")) # NEGATIVE
print(nlp("Je m'attendais à mieux de la part de Franck Dubosc !")) # NEGATIVE

# now predict input data
df = pd.read_csv("data/input/210115_Togo_networking_notes_formatted.csv", encoding='latin')
print("Dataframe size:", df.shape)

df_list = df.values
sentiments = []
scores = []

for index, row in enumerate(df_list):
    if index % 20 == 0:
        print(f'{(index/df.shape[0])*100:.0f}% done')
    result = nlp(row[1])
    sentiments.append(result[0]['label'])
    scores.append(result[0]['score'])

# write out results
df['label'] = sentiments
df['score'] = scores
df.to_csv("data/output/sentiment_analysis_results.csv", index=False)

