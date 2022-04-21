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

def show_elbow_method(canvas, data):

    modHardData = pd.read_csv('Granite1Normalized.csv')

    # Delete unneeded columns for our kmeans model
    del modHardData["X - Normalized"]
    del modHardData["Y - Normalized"]
    del modHardData["Hardness(HV)"]
    del modHardData["Test"]

    #converting dataset into np array to allow for slicing of the data set
    modHardData = np.array(data)

    wcss = []

    for i in range(1, 21):
        model = KMeans(n_clusters=i,
                    init='k-means++',  # Initialization method for kmeans
                    max_iter=300,  # Maximum number of iterations
                    n_init=10,  # Choose how often algorithm will run with different centroid
                    random_state=0)  # Choose random state for reproducibility
        model.fit(modHardData)
        wcss.append(model.inertia_)

    fig, ax = plt.subplots()
    ax.plot(range(1, 21), wcss)
    ax.set_title('Elbow Method')  # Set plot title
    ax.set_xlabel('Number of Clusters') # Set x axis name 
    ax.set_ylabel('Within Cluster Sum of Squares (WCSS)')
    #ax.xlabel('Number of clusters')  # Set x axis name
    #ax.ylabel('Within Cluster Sum of Squares (WCSS)')  # Set y axis name

    plt_canvas_agg = FigureCanvasTkAgg(fig, canvas)
    plt_canvas_agg.draw()
    plt_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return plt_canvas_agg

def show_kmeans(canvas, clusters, data, d):

    #converting dataset into np array to allow for slicing of the data set
    modHardData = np.array(data)

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
    if d:
        ax.scatter(modHardData[:, 0], modHardData[:, 1], c=pred_y, s=50, cmap='viridis')
        centers = kmeans.cluster_centers_
        ax.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
    else:
        ax.scatter(modHardData[:, 2], modHardData[:, 3], c=pred_y, s=50, cmap='viridis')
        centers = kmeans.cluster_centers_
        ax.scatter(centers[:, 2], centers[:, 3], c='black', s=200, alpha=0.5);
    """plt_canvas_agg = FigureCanvasTkAgg(fig, canvas)
    plt_canvas_agg.draw()
    plt_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return plt_canvas_agg"""
    return fig

def calculate_characteristic_values(useXY):
    modHardData = pd.read_csv('Granite1Normalized.csv')

    if not useXY:
        # Delete unneeded columns for our kmeans model
        del modHardData["X - Normalized"]
        del modHardData["Y - Normalized"]
        del modHardData["Hardness(HV)"]
        del modHardData["Test"]
    else: 
        del modHardData["Hardness(HV)"]
        del modHardData["Test"]

    df = pd.DataFrame([get_comparison_scores(k, modHardData) for k in range(2, 12)],
                  columns=['k', 'BIC', 'AIC', 'silhouette',
                           'davies', 'calinski'])
    return df;

def get_comparison_scores(idx, modHardData):
    kmeans = KMeans(n_clusters=idx,  # Set amount of clusters
                    init='k-means++',  # Initialization method for kmeans
                    max_iter=300,  # Maximum number of iterations
                    n_init=10,  # Choose how often algorithm will run with different centroid
                    random_state=0)
    kmeans.fit(modHardData)
    y_pred = kmeans.predict(modHardData)
    labels = kmeans.labels_
    bic, aic = get_bic_aic(idx, modHardData) # lower is better
    sil = silhouette_score(modHardData, y_pred) # higher is better
    db = davies_bouldin_score(modHardData, y_pred) # lower is better
    
    #Cannot run these as we don't have correct expected values 
    #hom = homogeneity_score(modHardData, y_pred)
    #com = completeness_score(modHardData, y_pred)
    #vms = v_measure_score(modHardData, y_pred)
    
    cal = calinski_harabasz_score(modHardData, y_pred) # higher is better
    return idx, bic, aic, sil, db, cal

def get_bic_aic(k, X):
    gmm = GaussianMixture(n_components=k, init_params='kmeans')
    gmm.fit(X)
    return gmm.bic(X), gmm.aic(X)

def plot_value(canvas, plotValue, df):
    
    fig, ax = plt.subplots()
    ax.plot(df['k'], df[plotValue])
    ax.set_title(f'{plotValue} vs clusterCount')
    ax.set_xlabel("Cluster Count")
    ax.set_ylabel(plotValue)

    plt_canvas_agg = FigureCanvasTkAgg(fig, canvas)
    plt_canvas_agg.draw()
    plt_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return plt_canvas_agg

def plot_scaled_difference(canvas, plotValue, df):
    r_min, r_max = df[plotValue].min(), df[plotValue].max()
    scaler = MinMaxScaler(feature_range=(r_min, r_max))

    fig, ax = plt.subplots()
    diff = np.abs(df[plotValue] - r_min)
    ax.plot(df['k'], diff)
    ax.set_title('Scaled Absolute Difference')
    ax.set_xlabel("Cluster Count")
    ax.set_ylabel('absolute difference')

    plt_canvas_agg = FigureCanvasTkAgg(fig, canvas)
    plt_canvas_agg.draw()
    plt_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return plt_canvas_agg

def plot_values(canvas, plotList, df):
    fig, ax = plt.subplots()
    ax.plot(df['k'], df[plotList[0]], color='tab:red')
    ax.set_title(f'{plotList[0]} and {plotList[1]}')
    ax.set_xlabel("Cluster Count")
    ax.set_ylabel(plotList[0], color='tab:red')
    ax.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax.twinx()
    ax2.plot(df['k'], df[plotList[1]], color='tab:blue')
    ax2.set_ylabel(plotList[1], color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    """for plotValue in plotList:
        ax.plot(df['k'], df[plotValue])
    ax.set_xlabel("Cluster Count")
    #ax.set_ylabel(plotValue)"""

    plt_canvas_agg = FigureCanvasTkAgg(fig, canvas)
    plt_canvas_agg.draw()
    plt_canvas_agg.get_tk_widget().pack(side="top", fill="both", expand=1)
    return plt_canvas_agg