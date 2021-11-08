import numpy as np 
np.set_printoptions(threshold=10000,suppress=True)  
#Optimisation de l'affichage, afficher que qques lignes
import pandas as pd #Lecture des fichiers csv,etc.
import warnings 
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore') #Eviter d'afficher des warnings
churn=pd.read_csv('./Churn_Modelling .csv',sep=',',header=0) 
churn.head(3) 
#Afficher les 3 premières lignes de la variable churn qui contient un dataFrame
#mettre ./ puis tab pour ouvrir la liste des fichiers dans le repertoire
#spécifier le separateur : , ; header =0 càd la première ligne contient les titres, 

X=churn.iloc[:,1:10].values
#on prend toutes les lignes, on va prendre de 1 à 10
Y=churn.iloc[:,10].values  
#pour les colonnes, la décomposition Que la 10ème colonne
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.25,random_state=1)
#La base de test contiendra 25% des indiviuds de la base totale.
#Importation des algorithmes 
from sklearn.tree import DecisionTreeClassifier #Arbre de decision
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score

from sklearn.neural_network import MLPClassifier #Reseaux de neurones 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import AdaBoostClassifier 
from sklearn.ensemble import ExtraTreesClassifier

MLP=MLPClassifier(hidden_layer_sizes=(20, 10), alpha=0.001, max_iter=200)
KNN=KNeighborsClassifier(n_neighbors=5)
RF=RandomForestClassifier(n_estimators=100, random_state=1)
Ada=AdaBoostClassifier(n_estimators=100, random_state=0)
ExtC=ExtraTreesClassifier(n_estimators=100, random_state=0)

def ClassDecisionTreeClassifier(Xtrain,Ytrain,Xtest,Ytest):
  print("DecisionTreeClassifier")
  print()
  DT=DecisionTreeClassifier(random_state=0, criterion='entropy') 
  DT.fit(Xtrain,Ytrain) #Apprentissage
  YDT=DT.predict(Xtest)
 # print('Accuracy = {0:.3f}'.format(accuracy_score(Ytest,YDT)))
  #print('Precision = {0:.3f}'.format(precision_score(Ytest,YDT)))
  #print('Recall = {0:.3f}'.format(recall_score(Ytest,YDT)))
  print('moyenee  = {0:.3f}'.format((recall_score(Ytest,YDT)+ accuracy_score(Ytest,YDT))/2))
  print()

def ClassKNeighborsClassifier(Xtrain,Ytrain,Xtest,Ytest):
  print("KNeighborsClassifier")
  print()
  KNN=KNeighborsClassifier(n_neighbors=5) 
  KNN.fit(Xtrain,Ytrain) #Apprentissage
  YDTE=KNN.predict(Xtest)
  #print('Accuracy = {0:.3f}'.format(accuracy_score(Ytest,YDTE)))
  #print('Precision = {0:.3f}'.format(precision_score(Ytest,YDTE)))
  #print('Recall = {0:.3f}'.format(recall_score(Ytest,YDTE)))
  print('moyenee  = {0:.3f}'.format((recall_score(Ytest,YDTE)+ accuracy_score(Ytest,YDTE))/2))
  print()
  
  
def ClassRandomForestClassifier(Xtrain,Ytrain,Xtest,Ytest):
  print("RandomForestClassifier")
  print()
  RF=RandomForestClassifier(n_estimators=100, random_state=1)
  RF.fit(Xtrain,Ytrain) #Apprentissage
  YDTR=RF.predict(Xtest)
 # print('Accuracy = {0:.3f}'.format(accuracy_score(Ytest,YDTR)))
 # print('Precision = {0:.3f}'.format(precision_score(Ytest,YDTR)))
 # print('Recall = {0:.3f}'.format(recall_score(Ytest,YDTR)))
  print('moyenee  = {0:.3f}'.format((recall_score(Ytest,YDTR)+ accuracy_score(Ytest,YDTR))/2))

  print()
  
  def ClassdaBoostClassifier(Xtrain,Ytrain,Xtest,Ytest):
    print("daBoostClassifier")
    print()
    Ada=AdaBoostClassifier(n_estimators=100, random_state=0)
    Ada.fit(Xtrain,Ytrain) #Apprentissage
    YDA=Ada.predict(Xtest)
    #print('Accuracy = {0:.3f}'.format(accuracy_score(Ytest,YDA)))
    #print('Precision = {0:.3f}'.format(precision_score(Ytest,YDA)))
    #print('Recall = {0:.3f}'.format(recall_score(Ytest,YDA)))
    print('moyenee  = {0:.3f}'.format((recall_score(Ytest,YDA)+ accuracy_score(Ytest,YDA))/2))
    

  print()
  def ClassExtraTreesClassifier(Xtrain,Ytrain,Xtest,Ytest):
    print("ExtraTreesClassifier")
    print()
    ExtC=ExtraTreesClassifier(n_estimators=100, random_state=0)
    ExtC.fit(Xtrain,Ytrain) #Apprentissage
    YDC=ExtC.predict(Xtest)
    #print('Accuracy = {0:.3f}'.format(accuracy_score(Ytest,YDC)))
    #print('Precision = {0:.3f}'.format(precision_score(Ytest,YDC)))
    #print('Recall = {0:.3f}'.format(recall_score(Ytest,YDC)))
    print('moyenee  = {0:.3f}'.format((recall_score(Ytest,YDC)+ accuracy_score(Ytest,YDC))/2))

  print()
  
  def ClassifierExcited(Xtrain,Ytrain,Xtest,Ytest):
    ClassDecisionTreeClassifier(Xtrain,Ytrain,Xtest,Ytest)
    ClassKNeighborsClassifier(Xtrain,Ytrain,Xtest,Ytest)
    ClassRandomForestClassifier(Xtrain,Ytrain,Xtest,Ytest)
    ClassdaBoostClassifier(Xtrain,Ytrain,Xtest,Ytest)
    ClassExtraTreesClassifier(Xtrain,Ytrain,Xtest,Ytest)
    
    
ClassifierExcited(Xtrain,Ytrain,Xtest,Ytest)


from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(Xtrain)
Xtrain_norme = ss.transform(Xtrain)
Xtest_norme = ss.transform(Xtest)

from sklearn.decomposition import PCA
pca =  PCA(n_components=3)
pca.fit(Xtrain_norme)
Xtrain_pca =pca.transform(Xtrain_norme)
Xtest_pca =pca.transform(Xtest_norme)
Xtrain_pca = np.concatenate((Xtrain_norme,Xtrain_pca),axis=1)
Xtest_pca = np.concatenate((Xtest_norme,Xtest_pca),axis=1)

from sklearn.model_selection import  GridSearchCV

parametres={'n_estimators':[100,200,300,400,500,600]}

model =AdaBoostClassifier()
GS = GridSearchCV(model,parametres, cv=5,scoring='accuracy')
GS.fit(Xtrain_pca,Ytrain)
GS.best_params_

from sklearn.pipeline import Pipeline 

import pickle  
from sklearn.pipeline import FeatureUnion


P = Pipeline([
              ('ss',StandardScaler()),
              ('FU',FeatureUnion([('ss',StandardScaler()),('pca',PCA(n_components=3))]) ),
              ('AD',AdaBoostClassifier(n_estimators=100))
])
P.fit(X,Y)
pickle.dump(P,open('classifieur.pkl','wb'))


