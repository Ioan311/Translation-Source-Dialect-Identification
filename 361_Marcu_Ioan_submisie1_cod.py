# importuri necesare
import os
import numpy as np
import matplotlib.pyplot as plt
# import pandas care citeste csv-uri 
import pandas as pd

# citire date 
data_path = '.'
train_data_df = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
test_data_df = pd.read_csv(os.path.join(data_path, 'test_data.csv'))

# codificare etichete in valori intregi de la 0 la 2
etichete_unice = train_data_df['label'].unique()
label2id = {}
id2label = {}
for idx, eticheta in enumerate(etichete_unice):
    label2id[eticheta] = idx
    id2label[idx] = eticheta

# aplicare dictionar label2id peste toate etichetele
labels = []
for eticheta in train_data_df['label']:
    labels.append(label2id[eticheta])
labels = np.array(labels)

# print(labels)

# preprocesarea datelor
import re

def proceseaza(text):
    # extrage semnele de punctuatie 
    text = re.sub("[-.,;:!?\"\'\/()_*=`]", "", text)
    # inlocuieste \n cu spatiu, face literele mici, strip() elimina caract. de la inceput sau de la sf de string 
    text = text.replace('\n', ' ').strip().lower()
    # split a string into a list 
    text_in_cuvinte = text.split(' ')
    return text_in_cuvinte

# aplicare functie de preprocesare pe intregul set de date
data = train_data_df['text'].apply(proceseaza)
# verificam daca afiseaza bine cu --> print(data)


# IMPARTIRE IN DATE DE ANTRENARE, VALIDARE, TEST
# 20% test, 15% validare(din 80%)
nr_test = int(20/100 * len(train_data_df))
# 8314

nr_ramase = len(data) - nr_test
nr_valid = int(15/100 * nr_ramase)
# 4988 

nr_train = nr_ramase - nr_valid
# 28268

# tot setul are --> print(len(data)) -> 41570 date
# luam indici de la 0 la 41569
indici = np.arange(0,len(train_data_df))
# dupa facem shuffle
np.random.shuffle(indici)

# impartirea se face in ordinea in care apar datele
train_data = data[indici[:nr_train]]
train_labels = labels[indici[:nr_train]]

valid_data = data[indici[nr_train : nr_train + nr_valid]]
valid_labels = labels[indici[nr_train : nr_train + nr_valid]]

test_data = data[indici[nr_train + nr_valid: ]]
test_labels = labels[indici[nr_train + nr_valid:]]

# Bag of Words -- numararea aparitiilor tuturor cuvintelor din date
from collections import Counter
# frecventa cuvintelor din setul de antrenare
counter = Counter()
# pt texte pre-procesate
for text in data:
    # fac update peste o lista de cuvinte si obtinem cat de frecvent apare fiecare cuv
    counter.update(text)

# se iau primele 10 cuv ca si caracteristici pt fiecare text 
N = 10
cuvinte_caracteristice = []
for cuvant, frecventa in counter.most_common(N):
    if cuvant.strip():
        cuvinte_caracteristice.append(cuvant)
# print(cuvinte_caracteristice)
cuvinte_caracteristice = [cuvant for cuvant, _ in counter.most_common(N)]

# fiecarui cuv ii atribuim un id in fct de pozitie 
# mapare intre cuvinte 
word2id = {}
id2word = {}
for idx, cuv in enumerate(cuvinte_caracteristice):
    word2id[cuv] = idx
    id2word[idx] = cuv
# print(word2id)
# print(id2word)


# numarare cuvinte text 
ctr = Counter(train_data.iloc[1])
# array de caracteristici 
features = np.zeros(len(cuvinte_caracteristice))
# bagam in array val din counter 
# fiecare pozitie din array trebuie sa reprezinte frecventa aceluiasi cuv in toate textele 
for idx in range(0, len(features)):
    # obtinem cuvantul pentru pozitia idx
    cuvant = id2word[idx]
    # asignam valoarea corespunzatoare frecventei cuvantului
    features[idx] = ctr[cuvant]
    # print('pt cuvantul ', cuvant, ' frecventa in textul 1 este ', ctr[cuvant])
# print([id2word[idx] for idx in range(0, len(features))]) -- vectorul de caracteristici 
# pe scurt am indexat fiecare cuv din cele 10 caracteristice si verificam vrecventa fiecaruia pentru fiecare text 

# functii pt Bag of Words
def count_most_common(how_many, texte_preprocesate):
    # Functie care returneaza cele mai frecvente cuvinte
    counter = Counter()
    for text in texte_preprocesate:
        # fac update peste o lista de cuvinte
        counter.update(text)
    cele_mai_frecvente = counter.most_common(how_many)
    cuvinte_caracteristice = [cuvant for cuvant, _ in cele_mai_frecvente]
    return cuvinte_caracteristice

def build_id_word_dicts(cuvinte_caracteristice):
    # Dictionarele word2id si id2word garanteaza o ordine pentru cuvintele caracteristice
    word2id = {}
    id2word = {}
    for idx, cuv in enumerate(cuvinte_caracteristice):
        word2id[cuv] = idx
        id2word[idx] = cuv
    return word2id, id2word

def featurize(text_preprocesat, id2word):
    # Pentru un text preprocesat dat si un dictionar care mapeaza pentru fiecare pozitie ce cuvant corespunde, returneaza un vector care reprezinta frecventele fiecarui cuvant.
    # numaram toate cuvintele din text
    ctr = Counter(text_preprocesat)
    # alocam un array care va reprezenta caracteristicile noastre
    features = np.zeros(len(id2word))
    # umplem array-ul cu valorile obtinute din counter, fiecare pozitie din array trebuie sa reprezinte frecventa aceluiasi cuvant in toate textele
    for idx in range(0, len(features)):
        # obtinem cuvantul pentru pozitia idx
        cuvant = id2word[idx]
        # asignam valoarea corespunzatoare frecventei cuvantului
        features[idx] = ctr[cuvant]

    return features

def featurize_multi(texte, id2word):
    # Pentru un set de texte preprocesate si un dictionar care mapeaza pentru fiecare pozitie ce cuvant corespunde, returneaza matricea trasaturilor tuturor textelor.
    all_features = []
    for text in texte:
        all_features.append(featurize(text, id2word))
    return np.array(all_features)

# transformare date in format vectorial
# luam primele 1000 de cuv caracteristice si frecvente 
cuvinte_caracteristice = count_most_common(1000, train_data)
word2id, id2word = build_id_word_dicts(cuvinte_caracteristice)

X_train = featurize_multi(train_data, id2word)
X_valid = featurize_multi(valid_data, id2word)
X_test = featurize_multi(test_data, id2word)

# modelul cu acuratetea cea mai buna este SVM   
from sklearn.metrics import accuracy_score
from sklearn import svm

model = svm.LinearSVC(C=0.5)

model.fit(X_train, train_labels)
vpreds = model.predict(X_valid)
tpreds = model.predict(X_test)

# print(accuracy_score(valid_labels, vpreds))
# print(accuracy_score(test_labels, tpreds))

# antrenam modelul cu SVM pe toate datele
toate_datele_vectorizate = featurize_multi(data, id2word)
model.fit(toate_datele_vectorizate, train_data_df['label'])

# Acum facem CROSS VALIDARE 
# esantionarea se face stratificata 
from sklearn.model_selection import StratifiedKFold, KFold
skf = StratifiedKFold(n_splits=5)
toate_etichetele = train_data_df['label'].values
print(skf.get_n_splits(toate_datele_vectorizate, toate_etichetele))
for train_index, test_index in skf.split(toate_datele_vectorizate, toate_etichetele):
    X_train_cv, X_test_cv = toate_datele_vectorizate[train_index], toate_datele_vectorizate[test_index]
    y_train_cv, y_test_cv = toate_etichetele[train_index], toate_etichetele[test_index]
    print(X_train_cv.shape)
    model = svm.LinearSVC(C=0.5)
    model.fit(X_train_cv, y_train_cv)
    tpreds = model.predict(X_test_cv)
    print(accuracy_score(y_test_cv, tpreds))

# Incepem pentru test_data.csv
# citim datele
test_data_df = pd.read_csv(os.path.join(data_path, 'test_data.csv'))

# preprocesam datele
date_test_procesate = test_data_df['text'].apply(proceseaza)

# aplicam Bag of Words pe datele pre-procesate
date_test_vectorizate = featurize_multi(date_test_procesate, id2word)

# obtinem predictii 
predictii = model.predict(date_test_vectorizate)

# Salvare predictii in csv
rezultat = pd.DataFrame({'id': np.arange(1, len(predictii)+1), 'label': predictii})

nume_fisier = ("submission.csv")

# salvare rezultat fara index
rezultat.to_csv(nume_fisier, index=False)