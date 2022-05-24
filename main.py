import sklearn.datasets as datasets
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from IPython.display import Image as PImage
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import seaborn as sb
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.preprocessing
import scipy.cluster.hierarchy as sch
import pylab
import sklearn.mixture as mixture
import pyclustertend 
import random
import numpy as np
from reader import Reader
import random
from graphviz import render

class main(object):

    def __init__(self, csvDoc):
        # Universal Doc
        self.csvDoc = csvDoc
        # Classes
        R = Reader(csvDoc)
        self.df = R.data
        self.significantCols = [" Net Income to Stockholder's Equity", " Quick Ratio",
                                " Interest Expense Ratio"," Current Liability to Equity",
                                " Cash/Total Assets"," Total debt/Total net worth",
                                " Persistent EPS in the Last Four Seasons"]

    def exploratory(self):
        print(self.df.shape)
        df = self.df.copy().groupby(["Bankrupt?"]).size()
        print(df)
    
    def kFolds(self):
        cv = KFold(n_splits=7) # Numero deseado de "folds" que haremos
        accuracies = list()
        max_attributes = len(list(self.df))
        depth_range = range(1, max_attributes + 1)
        
        # Testearemos la profundidad de 1 a cantidad de atributos +1
        for depth in depth_range:
            fold_accuracy = []
            tree_model = tree.DecisionTreeClassifier(criterion='entropy',
                                                    min_samples_split=20,
                                                    min_samples_leaf=5,
                                                    max_depth = depth,
                                                    class_weight={1:3.5})
            for train_fold, valid_fold in cv.split(self.df):
                f_train = self.df.loc[train_fold] 
                f_valid = self.df.loc[valid_fold] 
        
                model = tree_model.fit(X = f_train.drop(['Bankrupt?'], axis=1), 
                                    y = f_train["Bankrupt?"]) 
                valid_acc = model.score(X = f_valid.drop(['Bankrupt?'], axis=1), 
                                        y = f_valid["Bankrupt?"]) # calculamos la precision con el segmento de validacion
                fold_accuracy.append(valid_acc)
        
            avg = sum(fold_accuracy)/len(fold_accuracy)
            accuracies.append(avg)
        
        # Mostramos los resultados obtenidos
        df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
        df = df[["Max Depth", "Average Accuracy"]]
        print(df.to_string(index=False))
    
    def treeVisualization(self):
        y_train = self.df['Bankrupt?']
        x_train = self.df.drop(['Bankrupt?'], axis=1).values 
        
        # Crear Arbol de decision con profundidad = 3
        decision_tree = tree.DecisionTreeClassifier(criterion='entropy',
                                                    min_samples_split=20,
                                                    min_samples_leaf=5,
                                                    max_depth = 3,
                                                    class_weight={1:3.5})
        decision_tree.fit(x_train, y_train)
        
        # exportar el modelo a archivo .dot
        with open(r"tree1.dot", 'w') as f:
            f = tree.export_graphviz(decision_tree,
                                    out_file=f,
                                    max_depth = 7,
                                    impurity = True,
                                    feature_names = list(self.df.drop(['Bankrupt?'], axis=1)),
                                    class_names = ['No', 'Bankruptcy'],
                                    rounded = True,
                                    filled= True )
                                    
        return  decision_tree, x_train, y_train
    
    def precisionTree(self):
        decision_tree, x_train, y_train = self.treeVisualization()
        acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
        print(acc_decision_tree)

        
    def polRegression(self, status, grade, col1, col2):
        df =self.df.copy()
        df = df.drop(df[df["Bankrupt?"] == status].index) 
        x = df[col1]*100
        y = df[col2]*100

        mymodel = np.poly1d(np.polyfit(x, y, grade))
        myline = np.linspace(0, 100)
        plt.scatter(x, y, color='red')
        plt.xlabel(col1)
        plt.ylabel(col2)    
        plt.plot(myline, mymodel(myline))
        plt.show()
    

    def trainTest(self, col1, col2):
        df = self.df
        df = df[[col1,col2]]
        y = df.iloc[:,1:2].values#Dependent Vars
        X = df.iloc[:,0:1].values #Independent Var
        #Var Training
        X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3,train_size=0.7)

        return X, y, X_train, X_test, y_train, y_test, df
    
    # Build a linear model in order to watch out polynomial degree   
    def linearReg(self, X, y):
        lin_reg = LinearRegression()
        lin_reg.fit(X, y)
        return lin_reg

    def linearGraph(self, col1, col2):
        X, y, X_train, X_test, y_train, y_test, df = self.trainTest(col1, col2)
        linReg = self.linearReg(X,y)
        plt.scatter(X, y, color='red')
        plt.plot(X, linReg.predict(X), color='blue')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()
        
    def polyReg(self, X, y, deg):
        poly_reg = PolynomialFeatures(degree=deg)
        X_poly = poly_reg.fit_transform(X)
        pol_reg = LinearRegression()
        pol_reg.fit(X_poly, y)

        return poly_reg, pol_reg

    def polyGraph(self, degree, col1, col2):
        X, y, X_train, X_test, y_train, y_test, df = self.trainTest(col1, col2)
        polyReg, polReg = self.polyReg(X,y,degree)
        plt.scatter(X, y, color='red')
        plt.plot(X, polReg.predict(polyReg.fit_transform(X)), color='blue')
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.show()
    
    def hopkins(self):
        hop = 1
        df = self.df.copy()
        df = df[self.significantCols]
        df = df.dropna()
        Y = self.df["Bankrupt?"]
        self.X = np.array(df)
        self.Y = np.array(Y)
        X = self.X
        random.seed(123)
        df = df.reset_index()
        X_scale = sklearn.preprocessing.scale(X)
        df = df.reset_index()
        #hop = pyclustertend.hopkins(X,len(X))
        
        return X_scale
    
    def clusterNum(self):
        X_scale = self.hopkins()
        clustersNum = range(1,8)
        wcss = []
        for i in clustersNum:
            kmeans = cluster.KMeans(n_clusters=i)
            kmeans.fit(X_scale)
            wcss.append(kmeans.inertia_)

        plt.plot(clustersNum, wcss)
        plt.xlabel("Clusters")
        plt.ylabel("Score")
        plt.title("Clusters Amount")
        plt.show()
    
    def kMeansModel(self):
        X_scale = self.hopkins()
        kmeans = cluster.KMeans(n_clusters=5)
        y_kmeans = kmeans.fit_predict(self.X)
        X = self.X
        plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', label ='Cluster 1')
        plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', label ='Cluster 2')
        plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', label ='Cluster 3')
        plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', label ='Cluster 4')
        plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', label ='Cluster 5')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label = 'Centroids')
        plt.title('Clusters of Bankrupcy')
        plt.show()
        return 0   
        

col1 = " Persistent EPS in the Last Four Seasons"
col2 = " Interest Expense Ratio"
degree = 5
status = 0

reader = main('data.csv')
#reader.polRegression(status, degree, col1, col2)
#reader.exploratory()
#reader.kFolds()
#reader.precisionTree() 
#reader.linearGraph(col1, col2)
#reader.polyGraph(degree, col1, col2)
#reader.clusterNum()
reader.kMeansModel()

"""
self.significantCols = [" Net Income to Stockholder's Equity", " Quick Ratio",
                                " Interest Expense Ratio"," Current Liability to Equity",
                                " Cash/Total Assets"," Total debt/Total net worth",
                                " Persistent EPS in the Last Four Seasons"]
"""