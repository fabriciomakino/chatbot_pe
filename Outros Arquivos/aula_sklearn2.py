import pandas as pd
import numpy as np
from sklearn import tree

train = pd.read_csv('tabela_aula2.csv')    
y_train = train['Caro']
x_train = train.drop(['Caro'], axis=1).values 
decision_tree = tree.DecisionTreeClassifier(max_depth = 20, min_samples_leaf = 10, min_impurity_decrease = 0.0001)
decision_tree.fit(x_train, y_train)

with open("aula3.dot", 'w') as f:
     f = tree.export_graphviz(decision_tree,
                              out_file=f,
                              max_depth = 20,
                              impurity = True,
                              feature_names = list(train.drop(['Caro'], axis=1)),
                              class_names = ['False', 'True'],
                              rounded = True,
                              filled= True )
        
        
       