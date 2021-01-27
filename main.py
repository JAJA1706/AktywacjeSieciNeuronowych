#!/usr/bin/python3
import keras
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange
from IPython.display import clear_output
from neuralClass import Ukryta, ReLU, Logistyczna, Liniowa, TangensHiperboliczny, entropia_krzyzowa, entropia_krzyzowa_grad

def wczytajDane():
    (X_tren, Y_tren), (X_test, Y_test) = keras.datasets.mnist.load_data()
    X_tren = X_tren.astype(float) / 255.
    X_test = X_test.astype(float) / 255.    

    # z 60000 danych 10000 przeznaczamy na walidacje
    X_tren, X_val = X_tren[:-10000], X_tren[-10000:]
    Y_tren, Y_val = Y_tren[:-10000], Y_tren[-10000:]

    #splaszczamy rozmiar macierzy [50000, 28, 28] -> [50000, 784]
    X_tren = X_tren.reshape([X_tren.shape[0], -1])
    X_val = X_val.reshape([X_val.shape[0], -1])
    X_test = X_test.reshape([X_test.shape[0], -1])   

    return X_tren, Y_tren, X_val, Y_val, X_test, Y_test

X_tren, Y_tren, X_val, Y_val, X_test, Y_test = wczytajDane()

#ukladamy warstwy naszej sieci
siec = []
siec.append(Ukryta(X_tren.shape[1],100))
siec.append( TangensHiperboliczny() )
siec.append(Ukryta(100,200))
siec.append( TangensHiperboliczny() )
siec.append(Ukryta(200,10))

def doPrzodu(siec, X):
    #aktywujemy wszystkie warstwy sieci
    #i zapisujemy je w liscie
    aktywacje = []
    input = X
    for l in siec:
        aktywacje.append(l.doPrzodu(input))
        # przekazujemy wynik z ostatniej warstwy
        input = aktywacje[-1]
    
    assert len(aktywacje) == len(siec)
    return aktywacje

def przewiduj(siec,X):
    # szukamy i zwracamy indeks logita o najwiekszym prawdopodobienstwie
    logity = doPrzodu(siec,X)[-1]
    return logity.argmax(axis=-1)

def trenuj(siec,X,Y):
    # Trenujemy siec na danym Batchu X i Y

    aktywacje_warstw = doPrzodu(siec,X)
    wejscia_warstw = [X] + aktywacje_warstw
    logity = aktywacje_warstw[-1]
    
    # Obliczamy strate korzystajac z entropii krzyzowej z logitami
    strata = entropia_krzyzowa(logity,Y)
    strata_grad = entropia_krzyzowa_grad(logity,Y)
    
    # przechodzimy po sieciach od konca 
    # poniewaz korzystamy z metody wstecznej propagacji gradientu
    for idx in range(len(siec))[::-1]:
        warstwa = siec[idx]
        strata_grad = warstwa.doTylu(wejscia_warstw[idx],strata_grad)
        
    return np.mean(strata)

#W tej funkcji konwertujemy dane na mniejsze kawalki
def przejdz_po_minibatchach(wejscia, cele, ile_batchow):
    assert len(wejscia) == len(cele)
    idx = np.random.permutation(len(wejscia))
    for start_idx in trange(0, len(wejscia) - ile_batchow + 1, ile_batchow):
        fragment = idx[start_idx:start_idx + ile_batchow]
        yield wejscia[fragment], cele[fragment]

trening_log = []
val_log = []
for epoka in range(15): 
    for x_batch,y_batch in przejdz_po_minibatchach(X_tren, Y_tren, ile_batchow=32):
        trenuj(siec,x_batch,y_batch)
    
    trening_log.append( np.mean( przewiduj( siec,X_tren ) == Y_tren ) )
    val_log.append( np.mean( przewiduj( siec,X_val ) == Y_val ))
    clear_output()
    print("epoka",epoka)
    print("dokladnosc na zbiorze trenujacym:",trening_log[-1])
    print("dokladnosc na zbiorze walidacyjnym:",val_log[-1])
    plt.plot(trening_log,label='treningowy')
    plt.plot(val_log,label='walidacyjny')
    plt.legend(loc='best')
    plt.title("")
    plt.grid()
    plt.show()
    