---
title: "NLP: Classifying disaster tweets"
date: 2020-08-01
tags: [NLP, Kaggle]
excerpt: "Predict which Tweets are about real disasters and which ones are not"
mathjax: "true"
---
#NLP Disaster Tweet Kaggle Competition
Predict which Tweets are about real disasters and which ones are not

https://www.kaggle.com/c/nlp-getting-started


```python
#Import packages
import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = stopwords.words('english')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!
    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!


Read in the data


```python
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')
```


```python
pd.DataFrame({'count': train.target.value_counts(), 
              'percentage': train.target.value_counts(normalize=True)})
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4342</td>
      <td>0.57034</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3271</td>
      <td>0.42966</td>
    </tr>
  </tbody>
</table>
</div>




```python
train["target"].value_counts().plot(kind='bar')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f880e1c4908>




![png](output_5_1.png)


## Preprocess the data
Create a mapping function to turn each keyword into a number


```python
def map_keywords(series):
    mapper = {}
    u_series = series.unique()
    for i in range(len(u_series)):
        mapper[u_series[i]] = i
        
    return mapper

train_keyword_map = map_keywords(train.keyword)

train['keyword_num'] = train['keyword'].map(train_keyword_map)
test['keyword_num'] = test['keyword'].map(train_keyword_map)
```


```python
nltk.download('wordnet')
```

    [nltk_data] Downloading package wordnet to /root/nltk_data...
    [nltk_data]   Unzipping corpora/wordnet.zip.





    True



More processing for the `text` field


```python
def text_preprocessing(data):
    #remove whitespace and lower all words
    data = data.apply(lambda x: x.strip().lower())
    #replace digits
    data = data.apply(lambda x: re.sub(r'\d+', '', x))
    #replace punctuation
    data = data.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    #tokenize
    data = data.apply(lambda x : word_tokenize(x))
    #filter out stopwords
    data = data.apply(lambda x: [word for word in x if word not in stop_words])
    #remove inflection and return base word
    lemmatizer = WordNetLemmatizer()
    data = data.apply(lambda x: [lemmatizer.lemmatize(word, pos ='v') for word in x])
    #parts of speech tagging
    #data = data.apply(lambda x: [pos_tag(x)])
    return data

train['pro_text'] = text_preprocessing(train.text)
test['pro_text'] = text_preprocessing(test.text)
```


```python
vectorizer = TfidfVectorizer()
#joining words and fit transofrming
vector = vectorizer.fit_transform(["".join(i) for i in train["pro_text"]])
vector = vector.todense()
vector = np.concatenate((vector, np.reshape(np.array(train["keyword_num"]), (train.keyword.shape[0],-1))), axis=1)
print(vector.shape)

# vector_test = vectorizer.fit_transform(["".join(i) for i in test["text"]])
vector_test = vectorizer.transform(["".join(i) for i in test["pro_text"]])
vector_test = vector_test.todense()
vector_test = np.concatenate((vector_test, np.reshape(np.array(test["keyword_num"]), (test.keyword.shape[0],-1))), axis=1)
print(vector_test.shape)
```

    (7613, 8037)
    (3263, 8037)


Split the data into train and test sets


```python
xtrain, xtest, ytrain, ytest = train_test_split(vector, train['target'], train_size = 0.75)
```

## Train the BERT Model


```python
%%time
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
```

    CPU times: user 23.3 s, sys: 4.41 s, total: 27.7 s
    Wall time: 36.4 s



```python
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
```


```python
!pip install bert-tensorflow==1.0.1
from bert import tokenization
```

    Collecting bert-tensorflow==1.0.1
    [?25l  Downloading https://files.pythonhosted.org/packages/a6/66/7eb4e8b6ea35b7cc54c322c816f976167a43019750279a8473d355800a93/bert_tensorflow-1.0.1-py2.py3-none-any.whl (67kB)
    [K     |████████████████████████████████| 71kB 3.4MB/s 
    [?25hRequirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from bert-tensorflow==1.0.1) (1.15.0)
    Installing collected packages: bert-tensorflow
    Successfully installed bert-tensorflow-1.0.1



```python
tf.gfile = tf.io.gfile
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
```


```python
train_input = bert_encode([" ".join(i) for i in train.pro_text], tokenizer, max_len=160)
test_input = bert_encode([" ".join(i) for i in test.pro_text], tokenizer, max_len=160)
```


```python
def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])
    
    
    return model
```


```python
model = build_model(bert_layer, max_len=160)
model.summary()
```

    Model: "functional_1"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_word_ids (InputLayer)     [(None, 160)]        0                                            
    __________________________________________________________________________________________________
    input_mask (InputLayer)         [(None, 160)]        0                                            
    __________________________________________________________________________________________________
    segment_ids (InputLayer)        [(None, 160)]        0                                            
    __________________________________________________________________________________________________
    keras_layer (KerasLayer)        [(None, 1024), (None 335141889   input_word_ids[0][0]             
                                                                     input_mask[0][0]                 
                                                                     segment_ids[0][0]                
    __________________________________________________________________________________________________
    tf_op_layer_strided_slice (Tens [(None, 1024)]       0           keras_layer[0][1]                
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 1)            1025        tf_op_layer_strided_slice[0][0]  
    ==================================================================================================
    Total params: 335,142,914
    Trainable params: 335,142,913
    Non-trainable params: 1
    __________________________________________________________________________________________________



```python
from keras.callbacks import EarlyStopping
early = EarlyStopping(monitor='val_loss',mode='auto', baseline=None, restore_best_weights=False)
train_history = model.fit(
    train_input, train.target,
    validation_split=0.25,
    epochs=100,
    batch_size=16,
    callbacks=[early], verbose=1
)

model.save('model.h5')
```

    Epoch 1/100
    357/357 [==============================] - ETA: 0s - loss: 0.2179 - accuracy: 0.9163WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.1719s vs `on_test_batch_end` time: 0.4154s). Check your callbacks.


    WARNING:tensorflow:Callbacks method `on_test_batch_end` is slow compared to the batch time (batch time: 0.1719s vs `on_test_batch_end` time: 0.4154s). Check your callbacks.


    357/357 [==============================] - 629s 2s/step - loss: 0.2179 - accuracy: 0.9163 - val_loss: 0.4086 - val_accuracy: 0.8461
    Epoch 2/100
    357/357 [==============================] - 640s 2s/step - loss: 0.1399 - accuracy: 0.9490 - val_loss: 0.4532 - val_accuracy: 0.8456


## Make Predictions and Write Submission file


```python
predictions = model.predict(test_input)
arr = [1 if i>0.5 else 0 for i in predictions]


sample_sub=pd.read_csv('sample_submission.csv')
sub=pd.DataFrame({'id':sample_sub['id'].values.tolist(),'target':arr})
sub.to_csv('submission_bert.csv',index=False)
```
