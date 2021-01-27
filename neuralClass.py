from __future__ import print_function
import numpy as np
from math import exp

#ogolna klasa warstwy
class Warstwa:
    def __init__(self):
        pass
    
    def doPrzodu(self, wejscie):
        return wejscie
         
    def doTylu(self, wejscie, poprzedni_grad):
        jednostki = wejscie.shape[1]
        d_Warstwa_d_wejscie = np.eye(jednostki)
        return np.dot(poprzedni_grad, d_Warstwa_d_wejscie)

#warstwa ukryta
class Ukryta(Warstwa):
    def __init__(self, wejscie, wyjscie, tempo_uczenia=0.1):
        self.tempo_uczenia = tempo_uczenia
        
        #inicjalizacja wag
        self.waga = np.random.normal(loc=0.0, 
                                        scale = np.sqrt(2/(wejscie + wyjscie)), 
                                        size = (wejscie,wyjscie))
        self.bias = np.zeros(wyjscie)
        
    def doPrzodu(self,wejscie):
        return np.dot(wejscie,self.waga) + self.bias
    
    def doTylu(self,wejscie,poprzedni_grad):
        grad_wejscie = np.dot(poprzedni_grad, self.waga.T)

        waga_grad = np.dot(wejscie.T, poprzedni_grad)
        bias_grad = poprzedni_grad.mean(axis=0)*wejscie.shape[0]
        
        #macierze gradientow musze byc tego samego rozmiaru
        assert waga_grad.shape == self.waga.shape and bias_grad.shape == self.bias.shape
        
        #Tutaj wprowadzamy metode stochastycznego spadku. 
        self.waga = self.waga - self.tempo_uczenia * waga_grad
        self.bias = self.bias - self.tempo_uczenia * bias_grad
        
        return grad_wejscie

#funkcja straty
def entropia_krzyzowa(logity, oczekiwana_odpowiedz):
    logits_for_answers = logity[np.arange(len(logity)), oczekiwana_odpowiedz]
    entropia = - logits_for_answers + np.log(np.sum(np.exp(logity),axis=-1))
    
    return entropia
    
def entropia_krzyzowa_grad(logity, oczekiwana_odpowiedz):
    ones_for_answers = np.zeros_like(logity)
    ones_for_answers[np.arange(len(logity)), oczekiwana_odpowiedz] = 1
    softmax = np.exp(logity) / np.exp(logity).sum( axis=-1, keepdims=True )
    
    return (- ones_for_answers + softmax) / logity.shape[0]


                       #funkcje aktywacji
class Liniowa(Warstwa):
    def __init__(self):
        pass
    
    def doPrzodu(self, wejscie):
        return wejscie
    
    def doTylu(self, wejscie, poprzedni_grad):
        return poprzedni_grad

class Logistyczna(Warstwa):
    def __init__(self):
        pass
    
    def doPrzodu(self, wejscie):
        logistyczna_doPrzodu = 1 / (1 + np.exp( wejscie ))
        return logistyczna_doPrzodu
    
    def doTylu(self, wejscie, poprzedni_grad):
        logistyczna_grad = - np.divide(np.exp(wejscie), (1 + np.exp(wejscie))**2)
        return logistyczna_grad * poprzedni_grad


class ReLU(Warstwa):
    def __init__(self):
        pass
    
    def doPrzodu(self, wejscie):
        relu_doPrzodu = np.maximum(0,wejscie)
        return relu_doPrzodu
    
    def doTylu(self, wejscie, poprzedni_grad):
        relu_grad = wejscie > 0
        return poprzedni_grad*relu_grad

class TangensHiperboliczny(Warstwa):
    def __init__(self):
        pass
    
    def doPrzodu(self, wejscie):
        relu_doPrzodu = np.divide( (np.exp(wejscie*2) - 1), (np.exp(wejscie*2) + 1) )
        return relu_doPrzodu
    
    def doTylu(self, wejscie, poprzedni_grad):
        relu_grad = 1 / np.cosh(wejscie)**2
        return poprzedni_grad * relu_grad