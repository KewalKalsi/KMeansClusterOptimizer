import PySimpleGUI as sg
from Utility import show_elbow_method
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def show_kmeans_model():
    from Utility import show_kmeans

    modHardData = pd.read_csv('Granite1Normalized.csv')
    fullData = pd.read_csv('Granite1Normalized.csv')

    # Delete unneeded columns for our kmeans model
    del modHardData["X - Normalized"]
    del modHardData["Y - Normalized"]
    del modHardData["Hardness(HV)"]
    del modHardData["Test"]

    del fullData["Hardness(HV)"]
    del fullData["Test"]

    data = modHardData
    d = True;

    rowTwo = [sg.Button(f"Clusters: {num}") for num in range(1, 11)]
    elbow_window = sg.Window(
        title="KMeans", 
        layout = [
            [sg.Canvas(key="Kmeans", size=(960,720), expand_x=True, expand_y=True)],
            rowTwo, 
            [sg.Button("Quit"), sg.Button("Change Data")]
            ], 
        element_justification = 'c', 
        finalize=True, 
        modal = True, 
        resizable=True)
    kmeans_agg = show_kmeans(elbow_window["Kmeans"].TKCanvas, 3, data, d)
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
            kmeans_agg = show_kmeans(elbow_window["Kmeans"].TKCanvas, int(event[10:11]), data, d)
            plt_canvas_agg.get_tk_widget().forget()
            plt.close('all')
            plt_canvas_agg = FigureCanvasTkAgg(kmeans_agg, elbow_window["Kmeans"].TKCanvas)
            plt_canvas_agg.draw_idle()
            plt_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
            elbow_window.refresh()
        if event == "Change Data":
            if d is True:
                data = fullData
                d = False
            elif d is False:
                data = modHardData
                d = True
    elbow_window.close()


def show_characteristic_window():
    from Utility import calculate_characteristic_values
    from Utility import plot_value
    df = calculate_characteristic_values()
    Characteristics = ["CLUSTERS", "BIC", "AIC", "SILHOUETTE", "DAVIES", "CALINSKI"]
    buttonRow = [sg.Button(Characteristics[i]) for i in range(0,6)]
    characteristics_window = sg.Window(
        title="Characteristics", 
        layout = [
            buttonRow,
            [sg.Table(values=df.values.tolist(), headings = Characteristics, key="-data-")],
            [sg.Button("PLOT"), sg.Button("VS PLOT"), sg.Button("PLOT SCALED DIFFERENCE")]
        ], 
        element_justification = 'c', 
        modal = True, 
        resizable=True,
        size = (960,240)
    )
    plotValue = "CLUSTERS"
    while True:
        event, values = characteristics_window.read()

        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "CLUSTERS":
            df = df.sort_values(by=['k'], ascending = True)
        if event == "DAVIES":
            df = df.sort_values(by=['davies'], ascending = True)
            plotValue = "davies"
        if event == "AIC" or event == "BIC":
            df = df.sort_values(by=[event], ascending = True)
            plotValue = event
        if event == "SILHOUETTE":
            df = df.sort_values(by=['silhouette'], ascending = False)
            plotValue = "silhouette"
        if event == "CALINSKI":
            df = df.sort_values(by=['calinski'], ascending = False)
            plotValue = "calinski"
        if event == "PLOT" and plotValue != "CLUSTERS":
            plot_window = sg.Window(
                title="{plotValue} PLOT", 
                layout = [
                    [sg.Canvas(key="-PLOT-", size=(960,720))],
                    [sg.Button("Quit")]], 
                element_justification = 'c', 
                finalize=True, 
                modal = True)
            plot_value(plot_window["-PLOT-"].TKCanvas, plotValue)
            while True:
                event, values = plot_window.read()
                if event == "Exit" or event == sg.WIN_CLOSED:
                    break
                if event == "Quit":
                    break;    
            plot_window.close()
            
        # Update data on screen
        characteristics_window["-data-"].update(values = df.values.tolist())
        characteristics_window.refresh()
    characteristics_window.close()
    

modHardData = pd.read_csv('Granite1Normalized.csv')
fullData = pd.read_csv('Granite1Normalized.csv')

# Delete unneeded columns for our kmeans model
del modHardData["X - Normalized"]
del modHardData["Y - Normalized"]
del modHardData["Hardness(HV)"]
del modHardData["Test"]

del fullData["Hardness(HV)"]
del fullData["Test"]

data = modHardData

layout = [
    [sg.Text("How would you like to analyze clusters?", font = ("Arial, 20"))], 
    [sg.Button("Show KMeans Model")],
    [sg.Button("Elbow Method")], [sg.Button("Elbow Method - X,Y Included")], 
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
        show_elbow_method(elbow_window["-CANVAS-"].TKCanvas, modHardData)
        while True:
            event, values = elbow_window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            if event == "Quit":
                break;    
        elbow_window.close()
    if event == "Elbow Method - X,Y Included":
        elbow_window = sg.Window(title="Elbow Method - X,Y Included", layout = [[sg.Canvas(key="-CANVAS-", size=(960,720))],[sg.Button("Quit")]], element_justification = 'c', finalize=True, modal = True)
        show_elbow_method(elbow_window["-CANVAS-"].TKCanvas, fullData)
        while True:
            event, values = elbow_window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            if event == "Quit":
                break;    
        elbow_window.close()
    if event == "Characteristic Values of KMeans":
        show_characteristic_window()

window.close()


