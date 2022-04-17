import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, homogeneity_score, completeness_score, \
    v_measure_score, calinski_harabasz_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt

def get_bic_aic(k, X):
    gmm = GaussianMixture(n_components=k, init_params='kmeans')
    gmm.fit(X)
    return gmm.bic(X), gmm.aic(X)

def get_comparison_scores(idx, modHardData):
    kmeans = KMeans(n_clusters=idx,  # Set amount of clusters
                    init='k-means++',  # Initialization method for kmeans
                    max_iter=300,  # Maximum number of iterations
                    n_init=10,  # Choose how often algorithm will run with different centroid
                    random_state=0)
    kmeans.fit(modHardData)
    y_pred = kmeans.predict(modHardData)
    bic, aic = get_bic_aic(idx, modHardData) # lower is better
    sil = silhouette_score(modHardData, y_pred) # higher is better
    db = davies_bouldin_score(modHardData, y_pred) # lower is better
    #hom = homogeneity_score(modHardData, y_pred)
    #com = completeness_score(modHardData, y_pred)
    #vms = v_measure_score(modHardData, y_pred)
    cal = calinski_harabasz_score(modHardData, y_pred) # higher is better
    return idx, bic, aic, sil, db, cal

def clusters(df):
    df.sort_values(by=['k'], ascending = True)
    print(df)

def BIC(df):
    df.sort_values(by=['BIC'], ascending = True)
    print(df)

def AIC(df):
    df.sort_values(by=['AIC'], ascending = True)
    print(df)

def silhouette(df):
    df.sort_values(by=['silhouette'], ascending = False)
    print(df)

def davies(df):
    df.sort_values(by=['davies'], ascending = True)
    print(df)

def calinski(df):
    df.sort_values(by=['calinski'], ascending = False)
    print(df)

def plot_compare(df, y1, y2, x, fig, ax1):
    ax1.plot(df[x], df[y1], color='tab:red')
    ax1.set_title(f'{y1} and {y2}')
    ax1.set_xlabel(x)
    ax1.set_ylabel(y1, color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.plot(df[x], df[y2], color='tab:blue')
    ax2.set_ylabel(y2, color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

def plot_contrast(df, y1, y2, x, fig, ax):
    a = np.array(df[y1])
    b = np.array(df[y2])

    r_min, r_max = df[y1].min(), df[y1].max()
    scaler = MinMaxScaler(feature_range=(r_min, r_max))
    b = scaler.fit_transform(b.reshape(-1, 1))[:,0]

    diff = np.abs(a - b)
    ax.plot(df[x], diff)
    ax.set_title('Scaled Absolute Difference')
    ax.set_xlabel(x)
    ax.set_ylabel('absolute difference')

def plot_result(df, y1, y2, x):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    plot_compare(df, y1, y2, x, fig, axes[0])
    plot_contrast(df, y1, y2, x, fig, axes[1])
    plt.tight_layout()
    plt.show()

fullData = pd.read_csv('Granite1Normalized.csv')
modHardData = pd.read_csv('Granite1Normalized.csv')

del fullData["Modulus"]

# Delete unneeded columns for our kmeans model
del modHardData["X - Normalized"]
del modHardData["Y - Normalized"]
del modHardData["Hardness(HV)"]
del modHardData["Test"]

#print(modHardData)

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
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')  # Set plot title
plt.xlabel('Number of clusters')  # Set x axis name
plt.ylabel('Within Cluster Sum of Squares (WCSS)')  # Set y axis name
plt.show()

# Define base model
kmeans = KMeans(n_clusters = 3,                 # Set number of clusters
                init = 'k-means++',             # Using kmeans for initialization method
                max_iter = 300,                 # Maximum number of iterations
                n_init = 10,                    # How often the algorithm will run with different centroids 
                random_state = 0)               # Keeps randomness consistent across runs
# 
pred_y = kmeans.fit_predict(modHardData)

u_labels = np.unique(pred_y)

colors = {0:'tab:blue', 1:'tab:orange', 2:'tab:green', 'G':'tab:red', 3:'tab:purple', 4:'tab:brown', 5:'tab:pink'}

plt.scatter(modHardData[:, 0], modHardData[:, 1], c=pred_y, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.legend()
plt.show()

df = pd.DataFrame([get_comparison_scores(k, modHardData) for k in range(2, 12)],
                  columns=['k', 'BIC', 'AIC', 'silhouette',
                           'davies', 'calinski'])
i = "clusters";
while (i != "quit"):
    print("What would you like to sort by: ")
    i = input("#ofClusters, BIC, AIC, silhouette, davies, calinski\n")
    options = {"clusters" : clusters,
           "BIC" : BIC,
           "AIC" : AIC,
           "silhouette" : silhouette,
           "davies" : davies,
           "calinski" : calinski,
    }
    if (i != "quit"):
        options[i](df)

qualOne, qualTwo = "AIC", "BIC"
while (qualOne != "quit"):
    qualOne = input("Pick a characteristic to plot")
    qualTwo = input("And a second")
    plot_result(df, 'BIC', 'silhouette', 'k' )
