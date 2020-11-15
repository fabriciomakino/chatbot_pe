import pandas as pd
import numpy as np
from sklearn import tree
#from graphviz import Digraph

train = pd.read_csv('98074 - Aprendizado.csv')    
y_train = train['Caro']
x_train = train.drop(['Caro'], axis=1).values 
decision_tree = tree.DecisionTreeClassifier(min_impurity_decrease=0.0001, min_samples_leaf = 10, max_depth = 20)
decision_tree.fit(x_train, y_train)
with open("aula.dot", 'w') as f:
     f = tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 20,
                              impurity = True,
                              feature_names = list(train.drop(['Caro'], axis=1)),
                              class_names = ['False', 'True'],
                              rounded = True,
                              filled= True )        
   
        
       