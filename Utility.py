import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, homogeneity_score, completeness_score, \
    v_measure_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
matplotlib.use("TkAgg")

def show_elbow_method(canvas):

    modHardData = pd.read_csv('Granite1Normalized.csv')

    # Delete unneeded columns for our kmeans model
    del modHardData["X - Normalized"]
    del modHardData["Y - Normalized"]
    del modHardData["Hardness(HV)"]
    del modHardData["Test"]

    #converting dataset into np array to allow for slicing of the data set
    modHardData = np.array(modHardData)

    wcss = []

    for i in range(1, 11):
        model = KMeans(n_clusters=i,
                    init='k-means++',  # Initialization method for kmeans
                    max_iter=300,  # Maximum number of iterations
                    n_init=10,  # Choose how often algorithm will run with different centroid
                    random_state=0)  # Choose random state for reproducibility
        model.fit(modHardData)
        wcss.append(model.inertia_)

    # Show Elbow plot
    #plt.plot(range(1, 11), wcss)
    #plt.title('Elbow Method')  # Set plot title
    #plt.xlabel('Number of clusters')  # Set x axis name
    #plt.ylabel('Within Cluster Sum of Squares (WCSS)')  # Set y axis name
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), wcss)
    ax.set_title('Elbow Method')  # Set plot title
    #ax.xlabel('Number of clusters')  # Set x axis name
    #ax.ylabel('Within Cluster Sum of Squares (WCSS)')  # Set y axis name

    plt_canvas_agg = FigureCanvasTkAgg(fig, canvas)
    plt_canvas_agg.draw()
    plt_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return plt_canvas_agg
    #plt.show()

def show_kmeans(canvas, clusters):
    modHardData = pd.read_csv('Granite1Normalized.csv')

    # Delete unneeded columns for our kmeans model
    del modHardData["X - Normalized"]
    del modHardData["Y - Normalized"]
    del modHardData["Hardness(HV)"]
    del modHardData["Test"]

    #converting dataset into np array to allow for slicing of the data set
    modHardData = np.array(modHardData)

    #canvas.TKCanvas.delete("all")

    kmeans = KMeans(n_clusters = clusters,                 # Set number of clusters
                init = 'k-means++',             # Using kmeans for initialization method
                max_iter = 300,                 # Maximum number of iterations
                n_init = 10,                    # How often the algorithm will run with different centroids 
                random_state = 0)               # Keeps randomness consistent across runs
# 
    pred_y = kmeans.fit_predict(modHardData)

    u_labels = np.unique(pred_y)

    colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 'G':'tab:red', 3:'tab:purple', 4:'tab:brown', 5:'tab:pink'}

    fig, ax = plt.subplots()
    ax.scatter(modHardData[:, 0], modHardData[:, 1], c=pred_y, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    """plt_canvas_agg = FigureCanvasTkAgg(fig, canvas)
    plt_canvas_agg.draw()
    plt_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return plt_canvas_agg"""
    return fig
