import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import Canvas

Top = tk.Tk()
Top.title("Penguins-Classifier-Perceptron")
Top.geometry('1000x500')
Top.resizable(0,0)

canvas = Canvas(Top,width=500,height=500)
for x in range(5):
    canvas.create_line(x, 0, x, 500, fill="blue")
canvas.place(x=360,y=0)

Win_lbl = tk.Label(Top,text='Inputs',font=('Times New Roman',25),foreground = "#a52a2a")
Win_lbl.place(x=150, y=3)

Feat = ['bill_length_mm','bill_depth_mm','flipper_length_mm','gender','body_mass_g']
Feature1_lbl = tk.Label(Top,text='Select 1st Feature',font=('Verdana',12),foreground= "Blue")
Feature1_lbl.place(x=10, y=60)
Feature1 = ttk.Combobox(Top,width=15,font=('Verdana',10))
Feature1['values']=Feat
Feature1.current()
Feature1.place(x=200, y=60)

Feature2_lbl = tk.Label(Top,text='Select 2nd Feature',font=('Verdana',12),foreground= "Blue")
Feature2_lbl.place(x=10, y=100)
Feature2 = ttk.Combobox(Top,width=15,font=('Verdana',10))
Feature2['values']=Feat
Feature2.current()
Feature2.place(x=200, y=100)

Class = ['Adelie','Gentoo','Chinstrap']
Class1_lbl = tk.Label(Top,text='Select 1st Class',font=('Verdana',12),foreground= "Blue")
Class1_lbl.place(x=10, y=160)
Class1 = ttk.Combobox(Top, width=15,font=('Verdana',10))
Class1['values']=Class
Class1.current()
Class1.place(x=200, y=160)

Class2_lbl = tk.Label(Top,text='Select 2nd Class',font=('Verdana',12),foreground= "Blue")
Class2_lbl.place(x=10, y=200)
Class2 = ttk.Combobox(Top,width=15,font=('Verdana',10))
Class2['values']=Class
Class2.current()
Class2.place(x=200, y=200)

eta_lbl = tk.Label(Top,text='Enter eta value',font=('Verdana',12),foreground= "Blue")
eta_lbl.place(x=10, y=260)
eta = tk.Entry(Top,font=('Verdana',10),foreground= "Blue",width=10)
eta.place(x =200, y=260)

ebochs_lbl = tk.Label(Top,text='Enter no. of ebochs',font=('Verdana',12),foreground= "Blue")
ebochs_lbl.place(x=10, y=300)
ebochs = tk.Entry(Top,font=('Verdana',10),foreground= "Blue",width=10)
ebochs.place(x=200, y=300)

CheckBias=tk.IntVar()
Bias = tk.Checkbutton(Top,text="Bias",variable=CheckBias,onvalue=1,offvalue=0,font=('Verdana',20,'bold'),foreground= "Orange")
Bias.place(x=50, y=360)
def Train():
 if (Feature1.get() == Feature2.get()) or (Class1.get()==Class2.get()):
     messagebox.showerror(title="error",message="Please Donot reapet your choose ",parent=Top)
 else:
    print(Feature1.get())
    print(Feature2.get())
    print(Class1.get())
    print(Class2.get())
    print(ebochs.get())
    print(eta.get())
    print(CheckBias.get())
Modeling =tk.Button(Top,text="Predict",font=('Verdana',15,'bold'),bg = "#a52a2a",foreground = "white",command=Train)
Modeling.place(x=150,y=430)

Top.mainloop()
