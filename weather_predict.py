import pandas as pd
import numpy as np
from tkinter import *
import tkinter as tk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor   
from sklearn.metrics import accuracy_score 
from sklearn import metrics
from tkinter import font
from plotly.offline import iplot
import plotly as py
import plotly.tools as tls
import warnings
warnings.filterwarnings("ignore")



data = pd.read_csv('weather_update.csv')
##print(data)

data.isnull().any()

data = data.fillna(method='ffill')


X = data[['day','pressure', 'max_temp', 'min_temp', 'meandew', 'meanhum', 'meancloud', 'rainfall', 'population',
                 'sunshine_hour', 'wind_direction', 'wind_speed', 'air_health_quality']]
y = data['mean_temp']

#data split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   
rf= RandomForestRegressor(random_state=5, n_estimators=20)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

R2_reg_train = rf.score(X_train, y_train)
R2_reg_test = rf.score(X_test, y_test)

print('R squared for train data is: %.3f' % (R2_reg_train))
print('R squared for test data is: %.3f' % (R2_reg_test))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

import tkinter as tk

IMAGE_PATH = 'forecast.jpg'
WIDTH=900
HEIGTH = 850

root = tk.Tk()
root.title("Weather Temperature Prediction")


canvas = tk.Canvas(root, width=WIDTH, height=HEIGTH)
canvas.pack()
def chart():
    plt.figure(0)
    labels = ['yes', 'No']
    values = data['heat'].value_counts().values
    plt.pie(values, labels=labels, autopct='%1.0f%%')
    plt.title('Heat percentage(Yes or No)')

    plt.figure(1)
    labels = ['yes', 'No']
    values = data['wet'].value_counts().values

    plt.pie(values, labels=labels, autopct='%1.0f%%')
    plt.title('wet percentage(Yes or No)')


    plt.figure(2)
    labels = ['Less','none' ,'More', ]
    values = data['rain'].value_counts().values
    plt.pie(values, labels=labels, autopct='%1.0f%%')
    plt.title('rain percentage(Less, none,More)')
    plt.show()
button2 = tk.Button (root, text='Pie chart',command=chart, bg='#fcba03') # button to call the 'values' command above 
canvas.create_window(200, 700, window=button2)

def plot():
    # Display mean temp distribution
    plt.figure(3)
    data['mean_temp'].plot(kind = 'hist', title = 'mean temp Distribution')


    # Display  max temp distribution
    plt.figure(4)
    data['max_temp'].plot(kind = 'hist', title = 'max temp Distribution')


    # Display  mintemp distribution
    plt.figure(5)
    data['min_temp'].plot(kind = 'hist', title = 'min temp Distribution')
    plt.show()
button3 = tk.Button (root, text='Plot chart',command=plot, bg='#fcba03') # button to call the 'values' command above 
canvas.create_window(270, 700, window=button3)

def het():
    ###### data replacement###############
    e=data['heat'] = data['heat'].map({'YES': 1, 'NO': 0})
    print(e)

    # Display mean temp  distribution based on weather prediction
    plt.figure(6)
    sns.distplot(data[data['heat'] == 1]['mean_temp'], label='heat climate')
    sns.distplot(data[data['heat'] == 0]['mean_temp'], label = 'Not a heat climate')
    plt.xlabel('mean temp')
    plt.ylabel('Frequency')
    plt.title('mean temp  distribution based on weather prediction(heat)')
    plt.legend()

    # Display max temp  distribution based on weather prediction
    plt.figure(7)
    sns.distplot(data[data['heat'] == 1]['max_temp'], label='heat climate')
    sns.distplot(data[data['heat'] == 0]['max_temp'], label = 'Not a heat climate')
    plt.xlabel('max temp')
    plt.ylabel('Frequency')
    plt.title('max temp  distribution based on weather prediction(heat)')
    plt.legend()

    # Display min temp  distribution based on weather prediction
    plt.figure(8)
    sns.distplot(data[data['heat'] == 1]['min_temp'], label='heat climate')
    sns.distplot(data[data['heat'] == 0]['min_temp'], label = 'Not a heat climate')
    plt.xlabel('min temp')
    plt.ylabel('Frequency')
    plt.title('min temp  distribution based on weather prediction(heat)')
    plt.legend()
    plt.show()
button4 = tk.Button (root, text='DisPlot chart for het',command=het, bg='#fcba03') # button to call the 'values' command above 
canvas.create_window(380, 700, window=button4)

def wet():
    ###### data replacement###############
    
    data['wet'] = data['wet'].map({'YES': 0, 'NO': 1})

    # Display mean_temp  distribution based on weather prediction(wet)
    plt.figure(9)
    sns.distplot(data[data['wet'] == 1]['mean_temp'], label='Not a wet climate')
    sns.distplot(data[data['wet'] == 0]['mean_temp'], label = 'wet climate')
    plt.xlabel('mean temp')
    plt.ylabel('Frequency')
    plt.title('mean_temp  distribution based on weather prediction(wet)')
    plt.legend()

    # Display max temp  distribution based on weather prediction(wet)
    plt.figure(10)
    sns.distplot(data[data['wet'] == 1]['max_temp'], label='Not a wet climate')
    sns.distplot(data[data['wet'] == 0]['max_temp'], label = 'wet climate')
    plt.xlabel('max_temp')
    plt.ylabel('Frequency')
    plt.title('max temp  distribution based on weather prediction(wet)')
    plt.legend()

    # Display min temp  distribution based on weather prediction(wet)
    plt.figure(11)
    sns.distplot(data[data['wet'] == 1]['min_temp'], label='Not a wet climate')
    sns.distplot(data[data['wet'] == 0]['min_temp'], label = 'wet climate')
    plt.xlabel('min temp')
    plt.ylabel('Frequency')
    plt.title('min temp  distribution based on weather prediction(wet)')
    plt.legend()
    

    # Display min temp  distribution based on weather prediction(wet)
    plt.figure(12)
    sns.distplot(data[data['wet'] == 1]['min_temp'], label='Not a wet climate')
    sns.distplot(data[data['wet'] == 0]['min_temp'], label = 'wet climate')
    plt.xlabel('min temp')
    plt.ylabel('Frequency')
    plt.title('min temp  distribution based on weather prediction(wet)')
    plt.legend()
    plt.show()
button5 = tk.Button (root, text='DisPlot chart for wet',command=wet, bg='#fcba03') # button to call the 'values' command above 
canvas.create_window(520, 700, window=button5)

def rain():
    ###### data replacement###############
    data['rain'] = data['rain'].replace({'Less': 0, 'More': 1,'None':2})
    # Display min temp  distribution based on weather prediction(rain)
    plt.figure(13)
    sns.distplot(data[data['rain'] == 0]['min_temp'], label = ' less rain climate')
    sns.distplot(data[data['rain'] == 1]['min_temp'], label='heavy rain climate')
    sns.distplot(data[data['rain'] == 2]['min_temp'], label = 'Not rain climate')
    plt.xlabel('min temp')
    plt.ylabel('Frequency')
    plt.title('min temp  distribution based on weather prediction(rain)')
    plt.legend()


    # Display mean temp  distribution based on weather prediction(rain)
    plt.figure(14)
    sns.distplot(data[data['rain'] == 0]['mean_temp'], label = ' less rain climate')
    sns.distplot(data[data['rain'] == 1]['mean_temp'], label='heavy rain climate')
    sns.distplot(data[data['rain'] == 2]['mean_temp'], label = 'Not rain climate')
    plt.xlabel('mean temp')
    plt.ylabel('Frequency')
    plt.title('mean temp  distribution based on weather prediction(rain)')
    plt.legend()

    # Display mean temp  distribution based on weather prediction(rain)
    plt.figure(15)
    sns.distplot(data[data['rain'] == 0]['max_temp'], label = ' less rain climate')
    sns.distplot(data[data['rain'] == 1]['max_temp'], label='heavy rain climate')
    sns.distplot(data[data['rain'] == 2]['max_temp'], label = 'Not rain climate')
    plt.xlabel('max_temp')
    plt.ylabel('Frequency')
    plt.title('max_temp  distribution based on weather prediction(rain)')
    plt.legend()
    plt.show()
button6 = tk.Button (root, text='DisPlot chart for rain',command=rain, bg='#fcba03') 
canvas.create_window(650, 700, window=button6)

from PIL import Image, ImageTk

img = ImageTk.PhotoImage(Image.open(IMAGE_PATH).resize((WIDTH, HEIGTH), Image.ANTIALIAS))
canvas.background = img  # Keep a reference in case this code is put in a function.
bg = canvas.create_image(0, 0, anchor=tk.NW, image=img)

label0 = tk.Label(root, text='Days: ',width=20,font=("Times New Roman",10),bg='#389396')
canvas.create_window( 200, 50, window=label0)


label1 = tk.Label(root, text='Maximum of Temperature: ',width=20,font=("Times New Roman",10),bg='#389396')
canvas.create_window( 200, 100, window=label1)

label2= tk.Label(root, text='Minimum of Temperature: ',width=20,font=("Times New Roman",10),bg='#389396')
canvas.create_window( 200, 150, window=label2)


label3= tk.Label(root, text=' Mean dew point: ',width=20,font=("Times New Roman",10),bg='#389396')
canvas.create_window( 200, 200, window=label3)

label4= tk.Label(root, text=' Mean humidity: ',width=20,font=("Times New Roman",10),bg='#389396')
canvas.create_window( 200, 250, window=label4)

label5= tk.Label(root, text=' Mean pressure: ',width=20,font=("Times New Roman",10),bg='#389396')
canvas.create_window( 200, 300, window=label5)

label6= tk.Label(root, text=' Mean cloud: ',width=20,font=("Times New Roman",10),bg='#389396')
canvas.create_window( 200, 350, window=label6)

label7= tk.Label(root, text=' Mean rainfall: ',width=20,font=("Times New Roman",10),bg='#389396')
canvas.create_window( 200, 400, window=label7)

label8= tk.Label(root, text='  Population density: ',width=20,font=("Times New Roman",10),bg='#389396')
canvas.create_window( 200, 450, window=label8)

label9= tk.Label(root, text=' Number of sunshine hour: ',width=20,font=("Times New Roman",10),bg='#389396')
canvas.create_window( 200, 500, window=label9)

label10= tk.Label(root, text=' Mean wind direction: ',width=20,font=("Times New Roman",10),bg='#389396')
canvas.create_window( 200, 550, window=label10)

label11= tk.Label(root, text=' Mean wind speed: ',width=20,font=("Times New Roman",10),bg='#389396')
canvas.create_window( 200,600, window=label11)

label12= tk.Label(root, text=' Mean air health quality: ',width=20,font=("Times New Roman",10),bg='#389396')
canvas.create_window( 200,650, window=label12)

# create entry box

entry0 = tk.Entry (root, width=40) 
canvas.create_window(600, 50, window=entry0)

entry1 = tk.Entry (root, width=40) 
canvas.create_window(600, 100, window=entry1)

entry2 = tk.Entry (root, width=40) 
canvas.create_window(600, 150, window=entry2)

entry3 = tk.Entry (root, width=40) 
canvas.create_window(600, 200, window=entry3)

entry4 = tk.Entry (root, width=40) 
canvas.create_window(600, 250, window=entry4)

entry5 = tk.Entry (root, width=40) 
canvas.create_window(600, 300, window=entry5)

entry6 = tk.Entry (root, width=40) 
canvas.create_window(600, 350, window=entry6)

entry7 = tk.Entry (root, width=40) 
canvas.create_window(600, 400, window=entry7)

entry8 = tk.Entry (root, width=40) 
canvas.create_window(600, 450, window=entry8)

entry9 = tk.Entry (root, width=40) 
canvas.create_window(600, 500, window=entry9)

entry10 = tk.Entry (root, width=40) 
canvas.create_window(600, 550, window=entry10)

entry11 = tk.Entry (root, width=40) 
canvas.create_window(600, 600, window=entry11)


entry12 = tk.Entry (root, width=40) 
canvas.create_window(600, 650, window=entry12)

def values():

    global days #our 1st input variable
    days = float(entry0.get())
    
    global maximumTemperature #our 1st input variable
    maximumTemperature = float(entry1.get()) 
    
    global minimumtemperature #our 2nd input variable
    minimumtemperature = float(entry2.get())

    global meandewpoint #our 3st input variable
    meandewpoint= float(entry3.get())

    global meanhumidity #our 4st input variable
    meanhumidity = float(entry4.get())

    global meanpressure  #our 5st input variable
    meanpressure  = float(entry5.get())

    global meancloud #our 6st input variable
    meancloud = float(entry6.get())

    global meanrainfall#our 7st input variable
    meanrainfall= float(entry7.get())

    global populationdensity #our 8st input variable
    populationdensity= float(entry8.get())

    global numberofsunshinehour#our 9st input variable
    numberofsunshinehour = float(entry9.get())

    global meanwinddirection#our 10st input variable
    meanwinddirection =float(entry10.get())

    global  meanwindspeed#our 11st input variable
    meanwindspeed = float(entry11.get())

    global meanairhealthquality #our 12st input variable
    meanairhealthquality = float(entry12.get()) 
    
    Prediction_result  =  rf.predict([[days,maximumTemperature ,minimumtemperature,meandewpoint,meanhumidity,
                                                                           meanpressure,meancloud,meanrainfall ,
                                                                           populationdensity,numberofsunshinehour ,
                                                                           meanwinddirection,meanwindspeed ,
                                                                           meanairhealthquality ]])

    
    
    if Prediction_result >30:
        window = tk.Toplevel(root)
        window.configure(background='red')
        window.geometry("400x100")
        
        window.title("Temperature Checker")

        frame = tk.Frame(window,width=385, height=460,bg='red')
        frame.grid()
        msgbody1 = tk.Label(frame, bg="red",text="The", font=("Times New Roman", 20, "bold"))
        msgbody1.grid(row=1, column=1, sticky=N)
        lang = tk.Label(frame, bg="red",text="tempature is high", font=("Times New Roman", 20, "bold"), fg='blue')
        lang.grid(row=1, column=2, sticky=N)
        msgbody2 = tk.Label(frame, bg="red",text= Prediction_result, font=("Times New Roman", 20, "bold"))
        msgbody2.grid(row=1, column=3, sticky=N)

         
    else:
        win = tk.Toplevel(root)
        win.configure(background='green')
        win.geometry("400x100")
        
        win.title("Temperature Checker")

        frame = tk.Frame(win,width=385, height=460,bg='green')
        frame.grid()
        msgbody1 = tk.Label(frame, bg="green",text="The", font=("Times New Roman", 20, "bold"))
        msgbody1.grid(row=1, column=1, sticky=N)
        lang = tk.Label(frame, bg="green",text="tempature is low", font=("Times New Roman", 20, "bold"), fg='blue')
        lang.grid(row=1, column=2, sticky=N)
        msgbody2 = tk.Label(frame, bg="green",text= Prediction_result, font=("Times New Roman", 20, "bold"))
        msgbody2.grid(row=1, column=3, sticky=N)
         #tk.messagebox.showinfo('temperature',Prediction_result)        

button1 = tk.Button (root, text=' Temperature checker',command=values, bg='#fcba03') # button to call the 'values' command above 
canvas.create_window(100, 700, window=button1)

def climate():
    data = pd.read_csv('dataset.csv')

    data['Weather'] = data['Weather'].replace({'Sunny': 0, 'Rainy': 1,'Cold':2})



    X=data[['Temperature']]

    y=data['Weather']

##    plt.figure(16)
##    labels=['Sunny','Rainy','Cold']
##    values=data['Weather'].value_counts().values
##    plt.pie(values,labels=labels,autopct='%1.0f%%')
##    plt.title('Weather percentage(Sunny,Rainy,Cold)')
##    plt.show()
    from sklearn import metrics
    from tkinter import messagebox

    from sklearn.tree import DecisionTreeClassifier

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   
    rf= DecisionTreeClassifier(random_state=0)
    rf.fit(X_train, y_train)
    test=X_test.values.reshape(-1,1)
    y_pred = rf.predict(test)




    win1 = tk.Toplevel(root)
    win1.title("climate classfication")

    WIDTH=500
    HEIGTH = 300

    

    canvas = tk.Canvas(win1, width=WIDTH, height=HEIGTH)
    canvas.pack()

    label0 = tk.Label(win1, text='Temperature: ',font=("Times New Roman",10),bg='#389396')
    canvas.create_window( 50, 50, window=label0)

    entry0 = tk.Entry (win1, width=40) 
    canvas.create_window(220, 50, window=entry0)


    def values1():

        global Temperature #our 1st input variable
        Temperature = float(entry0.get())

        Prediction_result  =  rf.predict([[Temperature]])
        print(Prediction_result)
                                     
        if Prediction_result==0:
                tk.messagebox.showinfo('Climate','Sunny')
        elif Prediction_result==1:
                tk.messagebox.showinfo('Climate','Rainy')
        else:
               tk.messagebox.showinfo('Climate','Cold')


    button7 = tk.Button (win1, text=' Climate checker',command=values1, bg='#fcba03') # button to call the 'values' command above 
    canvas.create_window(150, 100, window=button7)
button8 = tk.Button (root, text=' climate classifier',command=climate, bg='#fcba03') # button to call the 'values' command above 
canvas.create_window(800, 700, window=button8)











    
