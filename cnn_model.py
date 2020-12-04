import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Embedding, Dense, Input, Conv1D, MaxPooling1D, Dropout, Concatenate, GlobalMaxPooling1D, Flatten
from tensorflow.keras.models import load_model, Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import lstm_model
from gensim.models import FastText
from gensim.models.keyedvectors import KeyedVectors
from gensim.utils import simple_preprocess

# ----------parameters section -------------
embedding_dim = 300
num_filters = 100
dropout = 0.5
filter_sizes = [3,4,5]

def CNNmodel(X_train, y_train, X_test, y_test,embedding_matrix, vocab_size):

    #CNN architecture
    
    maxlen = max(len(x) for x in X_train)

    #Shallow CNN
    #accuracy : 0.8480
    '''
    print("training CNN ...")
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen,weights=[embedding_matrix],trainable=False))
    model.add(Conv1D(512, 3, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    '''
    # Deep CNN
    # accuracy : 0.7660
    '''
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=maxlen,weights=[embedding_matrix],trainable=False))
    model.add(Conv1D(128, 7, activation='relu',padding='same'))
    model.add(MaxPooling1D())
    model.add(Conv1D(256, 5, activation='relu',padding='same'))
    model.add(MaxPooling1D())
    model.add(Conv1D(512, 3, activation='relu',padding='same'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    '''
    # Kim Yoon CNN
    # accuracy : 0.8240
    sequence_input = Input(shape=(max_len,), dtype='int32')

    embedding_layer = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=False)(sequence_input)
    embedding_layer_trainable = Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], input_length=max_len, trainable=True)(sequence_input)

    embedded_sequences = Concatenate()([embedding_layer, embedding_layer_trainable])
    #embedding_sequences = concate_embedding(sequence_input)
    #embedded_sequences = embedding_layer(sequence_input)

    convs = []
    #filter_sizes = [3,4,5]

    for fsz in filter_sizes:
        # Feature map size = (sentence_length - filter_size + 1) / 1
        # x = (None, (13,12,11), 100)
        x = Conv1D(num_filters, fsz, activation='relu',padding='valid')(embedded_sequences)
        # x = (None, (12,11,10), 100) padding이 없어서 한개씩 줄어들음
        x = MaxPooling1D(2, strides = 1,padding = 'valid')(x)
        convs.append(x)
        
    x = Concatenate(axis=1)(convs)
    x = Flatten()(x)
    #x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    output = Dense(6, activation='softmax')(x)
    model = Model(sequence_input, output)

    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    adam = Adam(lr = 1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('CNN_best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=32, validation_split=0.1)

    loaded_model = load_model('CNN_best_model.h5')
    print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))


def kerastokenizer(X_train, X_test, vocab_size):
    tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    #print(X_train[0])
    #print(tokenizer.word_index)

    return X_train, X_test, tokenizer.word_index

def load_pretrained(word_index, vocab_size, dir_path = '/home/hyejin/다운로드/wiki.ko/wiki.ko'):
    # 300 dimension
    # word2vec pretrained vector 사용할 때는 아래꺼 쓰기
    #ft = KeyedVectors.load_word2vec_format(dir_path, binary = True)
    # 원래는 위의 코드로 돌리려 시도했으나, pretrained model이 word2vec이 아닌 fasttext라서 위의 코드가 돌아가지 않았음.
    # fasttext binary 형식은 gensim word2vec과 다름, word2vec이 사용하지 않는 하위단어 단위에 대한 추가정보가 포함되기 때문
    # 하단의 3줄 코드처럼 돌려야 fasttext pretrained model을 사용할 수 있었음. 
    # 그냥 fasttext model을 다 로드하는 방법을 쓰면 너무 용량도 크고, 내가 원하는 key-value로 이루어지지 않아서 하단과 같은 방법을 사용.
    embedding_dict = KeyedVectors.load_word2vec_format(dir_path+'.vec', binary=False) 
    embedding_dict.save_word2vec_format(dir_path+"_gensim.bin", binary=True) 
    embedding_dict = KeyedVectors.load_word2vec_format(dir_path+"_gensim.bin", binary=True)

    #EMBEDDING_DIM = 300
    fixed_vector = np.random.normal(0,np.sqrt(0.25),embedding_dim)
    vocab_size2 = min(len(word_index)+1, vocab_size)
    embedding_matrix = np.zeros((vocab_size2, embedding_dim))
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        try:
            embedding_vector = embedding_dict[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = fixed_vector


    #embedding_layer = Embedding(vocab_size, EMBEDDING_DIM, weights=[embedding_matrix], traninable = True)
    
    return embedding_matrix, vocab_size2

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = lstm_model.open_file()

    max_len = 15
    total_token = lstm_model.count_token(X_train)
    # word_index 는 keras tokenizer를 통해 나온 set of words
    X_train, X_test, word_index = kerastokenizer(X_train, X_test, total_token)
    X_train, X_test = lstm_model.padding(X_train, X_test, max_len)
    y_train, y_test = lstm_model.convertLabel(y_train, y_test)

    embedding_matrix, vocab_size2 = load_pretrained(word_index, total_token)

    CNNmodel(X_train, y_train, X_test, y_test, embedding_matrix, vocab_size2)
