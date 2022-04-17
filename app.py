import PySimpleGUI as sg
from Utility import show_elbow_method
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt

def show_kmeans_model():
    from Utility import show_kmeans
    rowTwo = [sg.Button(f"Clusters: {num}") for num in range(1, 11)]
    elbow_window = sg.Window(title="KMeans", layout = [[sg.Canvas(key="Kmeans", size=(960,720), expand_x=True, expand_y=True)],rowTwo, [sg.Button("Quit")]], element_justification = 'c', finalize=True, modal = True, resizable=True)
    kmeans_agg = show_kmeans(elbow_window["Kmeans"].TKCanvas, 3)
    plt_canvas_agg = FigureCanvasTkAgg(kmeans_agg, elbow_window["Kmeans"].TKCanvas)
    plt_canvas_agg.draw()
    plt_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    while True:
        event, values = elbow_window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Quit":
            break;
        if event[0:8] == "Clusters":
            elbow_window["Kmeans"].TKCanvas.delete("all")
            kmeans_agg = show_kmeans(elbow_window["Kmeans"].TKCanvas, int(event[10:11]))
            plt_canvas_agg.get_tk_widget().forget()
            plt.close('all')
            plt_canvas_agg = FigureCanvasTkAgg(kmeans_agg, elbow_window["Kmeans"].TKCanvas)
            plt_canvas_agg.draw_idle()
            plt_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
            elbow_window.refresh()
            #elbow_window.refresh()
    elbow_window.close()

layout = [
    [sg.Text("How would you like to analyze clusters?", font = ("Arial, 20"))], 
    [sg.Button("Show KMeans Model")],
    [sg.Button("Elbow Method")], 
    [sg.Button("Characteristic Values of KMeans")]
]


window = sg.Window(title="Ideal Cluster Collecter", layout = layout, margins=(320, 240), element_justification = 'c', finalize=True)

while True:
    event, values = window.read()
    # End program if user closes window or
    # presses the OK button
    if event == "OK" or event == sg.WIN_CLOSED:
        break
    if event == "Show KMeans Model":
        show_kmeans_model();
    if event == "Elbow Method":
        elbow_window = sg.Window(title="Elbow Method", layout = [[sg.Canvas(key="-CANVAS-", size=(960,720))],[sg.Button("Quit")]], element_justification = 'c', finalize=True, modal = True)
        show_elbow_method(elbow_window["-CANVAS-"].TKCanvas)
        while True:
            event, values = elbow_window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            if event == "Quit":
                break;    
        elbow_window.close()
    if event == "Characteristic Values of KMeans":
        print("Characteristic Values of Kmeans Chosen ")

window.close()


