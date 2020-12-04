import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

def lstm(vocab_size, X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Embedding(vocab_size, 100))
    model.add(LSTM(64))
    model.add(Dense(6, activation='softmax'))

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    #history = model.fit(X_train, y_train, epochs=15, batch_size=64)
    history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=32, validation_split=0.1)
    
    #loss, accuracy = model.evaluate(X_test, y_test, batch_size=1)
    loaded_model = load_model('best_model.h5')
    print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))
    #return loss, accuracy

def padding(X_train, X_test, max_len):
    X_train = pad_sequences(X_train, maxlen=max_len)
    X_test = pad_sequences(X_test, maxlen=max_len)
    
    return X_train, X_test

def kerastokenizer(X_train, X_test, vocab_size):
    tokenizer = Tokenizer(vocab_size, oov_token = 'OOV') 
    tokenizer.fit_on_texts(X_train)
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    #print(X_train[0])
    #print(tokenizer.word_index)

    return X_train, X_test

def show_lengthOfSequence(X_train):
    print('최대 길이 :',max(len(l) for l in X_train)) #29
    print('평균 길이 :',sum(map(len, X_train))/len(X_train)) #7.27
    plt.hist([len(s) for s in X_train], bins=50)
    plt.xlabel('length of samples')
    plt.ylabel('number of samples')
    plt.show() #15가 좋을듯
    
def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if(len(s) <= max_len):
        cnt = cnt + 1
  print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

def count_token(X_train):
    total_cnt = 0
    for _line in X_train:
        total_cnt += len(_line)
    #avg_len = total_cnt/len(X_train)

    return total_cnt

def convertLabel(y_train, y_test):
    label_dict = {'ABBR\n':0, 'ENTY\n':1, 'DESC\n':2, 'HUM\n':3, 'LOC\n':4, 'NUM\n':5}
    y_train = [label_dict[a] for a in y_train]
    y_test = [label_dict[a] for a in y_test]

    y_train = to_categorical(y_train, num_classes = 6)
    y_test = to_categorical(y_test, num_classes = 6)    

    return y_train, y_test

def open_file(directory = './data/'):
    X_train = []
    y_train = []

    xt = open(directory+'kma_text_trans_re_train_5500.txt','r')
    xtl = xt.readlines()
    X_train = [_a.split() for _a in xtl]
    xl = open(directory+'label_trans_re_train_5500.label','r')
    y_train = xl.readlines()

    X_test = []
    y_test = []
    
    xt = open(directory+'kma_text_trans_re_TREC_10.txt','r')
    xtl = xt.readlines()
    X_test = [_a.split() for _a in xtl]
    xl = open(directory+'label_trans_re_TREC_10.label','r')
    y_test = xl.readlines()

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = open_file()
    #show_lengthOfSequence(X_train)
    #below_threshold_len(15, X_train)
    
    max_len = 15
    vocab_size = count_token(X_train)
    print('vocab size = ',vocab_size)
    
    X_train, X_test = kerastokenizer(X_train, X_test, vocab_size)
    '''
    X_train, X_test = padding(X_train, X_test, max_len)
    y_train, y_test = convertLabel(y_train, y_test)
    #print(len(y_train), y_train[0])
    lstm(vocab_size, X_train, y_train, X_test, y_test)
    #print(loss, acc)
    '''