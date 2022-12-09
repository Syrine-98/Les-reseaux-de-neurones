#Importation de seaborn
import seaborn as sns 

#Importation de matplot
import matplotlib.pyplot as plt

#Importation de la bibliothèque de division
from sklearn import model_selection
from sklearn.model_selection import train_test_split

#Importation de MLPClassifier
from sklearn.neural_network import MLPClassifier

#Importation de metrics
from sklearn import metrics

#Importation de pp_matrix
from pretty_confusion_matrix import pp_matrix_from_data

#Importation de keras et TensorFlow
import keras 
from sklearn.preprocessing import normalize 
from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
from tensorflow.keras.layers import (BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense)
from keras.utils import np_utils

#Importation de numpy
import numpy as np

#Importation des données
import pandas as pd
dataset = pd.read_csv("Iris.csv",header=0)

#Affichage des 10 premières lignes
print("\n")
print("Les 10 premières lignes du DataFrame : ")
print(dataset.head(10))

#Affichage des dimensions du Dataframe
print("\n")
print("Les dimensions du Dataframe sont : ")
print(dataset.shape)


#Visualisation des données en fonction de la longueur des pétales et de largeur des sépales
sns.pairplot(data = dataset , vars=('PetalLengthCm' , 'SepalWidthCm'),
hue = 'Species')
plt.show()

#Labellisation des différentes espèces d’iris
df = dataset.replace({'Species' : {'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2 }}, regex=True)

#Affichage des 10 premières lignes
print("\n")
print("Les 10 premières lignes du DataFrame après labellisation: ")
print(df.head(10))

#Division de dataset en des données d’apprentissage (70%) et des données de test (30%)
train , test = model_selection.train_test_split(df, test_size = 0.3)
trainX = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
trainY = train.Species
testX = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
testY = test.Species

#Affichage des 10 premières données d’apprentissage et celles de test
print("\n")
print("\n Les 10 premières données d’apprentissage sont : ")
print(trainX.head(10), '\n\n', trainY.head(10))
print("\n")
print("Les 10 premières données de test sont:")
print(testX.head(10), '\n\n', testY.head(10))

#Utilisation d'un perceptron multicouches pour l’apprentissage des données 
#Perceptron avec : un optimisateur (‘lbfgs’, epsilon=0.07 et nombre maximum d’itération=150)
clf = MLPClassifier(solver='lbfgs',alpha=1e-5, epsilon=0.07, max_iter=150)
clf.fit(trainX, trainY)

#Evaluation du perceptron en affichant son “accuracy” et le temps de réponse
#Affichage de l'accuracy
prediction=clf.predict(testX)
print("\n")
print("Les valeurs de prédiction sont : \n")
print(prediction)
print("\n")
print("Les valeurs de test sont : \n" )
print(testY.values)
print("\n")
print('L accuracy de ce perceptron multicouches est :',metrics.accuracy_score(prediction ,testY))

#Affichage de temps de réponse pour l'apprentissage
train_score = clf.score(trainX, trainY)
print("\n")
print("Le temps de réponse pour l'apprentissage est {}".format(train_score))

#Affichage de temps de réponse pour le test
test_score = clf.score(testX, testY)
print("\n")
print("Le temps de réponse pour le test est {}".format(test_score))
print("\n")

#Affichage de la matrice de confusion 
cmap='PuRd'
pp_matrix_from_data(testY.values, prediction)

#Ajout du paramètre de taux d’apprentissage au niveau du classifieur utilisé pour une valeur égale 0.7
clf = MLPClassifier(solver='sgd',alpha=1e-5, epsilon=0.07, max_iter=150,learning_rate_init=0.7)
clf.fit(trainX, trainY)

plt.plot(clf.loss_curve_)
plt.title("L'évolution d'apprentissage : Taux d'apprentissage = 0.7 ",fontsize=14)
plt.show()

clf.fit(testX, testY)
plt.plot(clf.loss_curve_)
plt.title("L'évolution de test : Taux d'apprentissage = 0.7 ",fontsize=14)
plt.show()

#Etude de la variation du paramètre de taux d’apprentissage 
#Changement du taux d'apprentissage
params= [
    {
        "solver":"sgd",
        "learning_rate":"constant",
        "learning_rate_init":0.2,
        "max_iter":150,
    },
      {
        "solver":"sgd",
        "learning_rate":"constant",
        "learning_rate_init":0.7,
        "max_iter":300,
    },
  {
        "solver":"sgd",
        "learning_rate":"invscaling",
        "learning_rate_init":0.2,
        "max_iter":300,
    },
      {
        "solver":"sgd",
        "learning_rate":"invscaling",
        "learning_rate_init":0.7,
        "max_iter":150,
    },
    {
       "solver" :"adam",
       "learning_rate_init":0.01,
      "max_iter":300,
    },
]
labels= [
    "constant learning-rate_0.2",
    "constant learning-rate_0.7",
    "invscaling learning-rate_0.2",
    "invscaling learning-rate_0.7",
    "adam",

]
plot_args = [
    {"c":"red","linestyle":"-"},
    {"c":"green","linestyle":"-"},
    {"c":"blue","linestyle":"-"},
    {"c":"red","linestyle":"--"},
    {"c":"green","linestyle":"--"},
]

#Affichage des scores d'apprentissage
mlps1=[]

for label,param in zip(labels,params):
  print("Apprentissage : %s " % label)
  clf=MLPClassifier(random_state=0,**param)
  clf.fit(trainX,trainY)
  mlps1.append(clf)
  print("Training score : %f " % clf.score(trainX,trainY))
  print("\n")
  
#Affichage des scores de test
mlps2=[]
for label,param in zip(labels,params):
  print("Test : %s" % label)
  clf.fit(testX,testY)
  mlps2.append(clf)
  print("Training set score : %f" % clf.score(testX,testY))
  print("\n")
   
#Affichage des courbes d'évolution d’apprentissage en fonction de variation du taux d’apprentissage
for mlp1, label, args in zip(mlps1,labels,plot_args):
  plt.plot(mlp1.loss_curve_ , 'g', label=label)
  plt.title("Apprentissage : %s \n" % label,fontsize=14)
  plt.show()

#Affichage des courbes d'évolution de test en fonction de variation du taux d’apprentissage
for mlp2, label, args in zip(mlps2,labels,plot_args):
  plt.plot(mlp2.loss_curve_ , 'r', label=label)
  plt.title("Test : %s \n" %label,fontsize=14)
  plt.show()
  
#Fixation du nombre d’itération égale à 10 fois le nombre fixé au début(150)
clf = MLPClassifier(solver='sgd',alpha=1e-5, epsilon=0.07, max_iter=1500,learning_rate_init=0.7)
clf.fit(trainX, trainY)

plt.plot(clf.loss_curve_)
plt.title("L'évolution d'apprentissage : Taux d'apprentissage = 0.7 et 1500 itérations",fontsize=14)
plt.show()

#Test des autres classifieurs de type réseau de neurones
data=df.iloc[np.random.permutation(len(df))]
X=data.iloc[:,1:5].values
y=data.iloc[:,5].values

#Normalisation
X_normalized=normalize(X,axis=0)
total_length=len(data)
train_length=int(0.8*total_length)
test_length=int(0.2*total_length)
trainX=X_normalized[:train_length]
testX=X_normalized[train_length:]
trainY=y[:train_length]
testY=y[train_length:]

trainY=np_utils.to_categorical(trainY,num_classes=3)
testY=np_utils.to_categorical(testY,num_classes=3)

model=Sequential()
model.add(Dense(1000,input_dim=4,activation='relu'))
model.add(Dense(500,activation='relu'))
model.add(Dense(300,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(trainX,trainY,validation_data=(testX,testY),batch_size=20,epochs=10)

#Evaluation : Affichage de l'accuracy 
prediction=model.predict(testX)
length=len(prediction)
y_label=np.argmax(testY,axis=1)
predict_label=np.argmax(prediction,axis=1)
accuracy=np.sum(y_label==predict_label)/length 
print("\n L'accuracy est : ",accuracy )