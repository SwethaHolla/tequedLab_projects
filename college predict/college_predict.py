from tkinter import *
#import tkinter from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
admissions = pd.read_csv("marks_data.csv")
print(admissions.head())

feature_cols = ['entrance','percentage']
X = admissions[feature_cols] # Features
y = admissions.admit # Target variable

ent_min = admissions['entrance'].min()
ent_max = admissions['entrance'].max()
admissions['ent_norm'] = (admissions['entrance'] - ent_min) / (ent_max - ent_min)
admissions.head()

per_min = admissions['percentage'].min()
per_max = admissions['percentage'].max()
admissions['per_norm'] = (admissions['percentage'] - per_min) / (per_max - per_min)
admissions.head()

def predict():
    
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics

    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()

    # fit the model with data
    #logreg.fit(X_train,y_train)
    from sklearn import linear_model
    lr = linear_model.LogisticRegression()
    lr.fit(X_train, y_train)
    mul_lr = linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X_train, y_train)
    #
    y_pred=mul_lr.predict(X_test)
    print(X_test)
    print(y_pred)
    y_pred1=mul_lr.predict([[496.273937,3.691417]])
    print("pred",y_pred1)

    entr=ent1.get()
    perc=ent2.get()
    #entr=input("Enter the Entrance exam score")
    #perc=input("Enter the percentage")
    #entr=entr.astype(np.float64)
    #perc=perc.astype(np.float64)
    e=float(entr)
    p=float(perc)
    print(type(e))
    if e>60:
        c='Invalid input'
        t1.delete("1.0", END)           #t1=tab1 consists of disease predicted by this alg   #previous value is deleted
        t1.insert(END, c)
        exit()
        
    elif p>100:
        c='Invalid input'
        t1.delete("1.0", END)           #t1=tab1 consists of disease predicted by this alg   #previous value is deleted
        t1.insert(END, c)
        exit()
      
    y1=mul_lr.predict([[e,p]])
    print(y1)
    print("Accuracy : ", metrics.accuracy_score(y_test,y_pred))
    if y1==[1]:
        c='RVCE'
    elif y1==[2]:
        c='BIT'
    elif y1==[3]:
        c='BNMIT'
    elif y1==[4]:
        c='SJBIT'
    elif y1==[5]:
        c='KSIT'
    elif y1==[6]:
        c='Global'
    elif y1==[7]:
        c='JIT'
    elif y1==[8]:
        c='New Horizon'
    elif y1==[9]:
        c='AMC'
    elif y1==[10]:
        c='East West'
    from sklearn import metrics
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    print(cnf_matrix)
    t1.delete("1.0", END)           #t1=tab1 consists of disease predicted by this alg   #previous value is deleted
    t1.insert(END, c)


root = Tk()
root.configure(background='khaki2')
#root.geometry("900*600")
t1 = Text(root, height=1, width=20,bg="cyan",fg="black")  #textbox to display disease predicted
t1.grid(row=9, column=4, padx=10)
    # Heading
w2 = Label(root, justify=CENTER, text="COLLEGE PREDICTOR", fg="black",bg="steel blue")
w2.config(font=("Times New Roman", 35))
w2.grid(row=1, column=3,columnspan=2, padx=100)            #dividing window into grids usng rows and cols
#w2 = Label(root, justify=LEFT, text="", fg="white", bg="blue")
#w2.config(font=("Aharoni", 30))
#w2.grid(row=2, column=0, columnspan=2, padx=100)    #padx is distance from x-axis i.e. horizontal dist

    # labels
  

S1Lb = Label(root, text="Entrance", fg="yellow", bg="black")
S1Lb.grid(row=7, column=2, pady=10, sticky=W) #sticky is similar to anchors

S2Lb = Label(root, text="Percentage", fg="yellow", bg="black")
S2Lb.grid(row=8, column=2, pady=10, sticky=W)

S3Lb = Label(root, text="College Predicted", fg="yellow", bg="black")
S3Lb.grid(row=9, column=2, pady=10, sticky=W)


    # entries
ent1=Entry(root)
ent1.grid(row=7,column=4)

ent2=Entry(root)
ent2.grid(row=8,column=4)

#txt=Text(root, width="30",height="10")
#txt.grid(row=8, column=5)


dst = Button(root, text="Predict", command=predict,bg="green",fg="yellow")    #command calls the decisiontree function
dst.grid(row=10, column=3,padx=10,pady=15)

    #textfileds
#t1 = Text(root, height=1, width=40,bg="orange",fg="black")  #textbox to display disease predicted
#t1.grid(row=15, column=1, padx=10)

#t2 = Text(root, height=1, width=40,bg="orange",fg="black")
#t2.grid(row=17, column=1 , padx=10)
root.mainloop()
