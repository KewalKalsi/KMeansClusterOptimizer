## KMeans Model Cluster Optimizer

This is a simple application utilizing PySimpleGUI. The functionality within the app is based off of two nanoindentation data sets, one that contains just Modulus and Hardness values and the other that contains Modulus, Hardness, as well as X and Y coordinates. 

The app supports plotting the fit data set over different cluster counts, plotting the elbow method to determine optimal cluster count, and analyzing/plotting specific characteristic values for each cluster count.

### How To Run 

To run the app with GUI, use the command:  
`python app.py`  

To run the app through text commands (which may not be as up-to-date as the GUI), use the command:   
`python main.py`  

Note - Depending on you're python install "python" may have to be replaced by "python3" when running the commands above.    

### TO-DO   

1. Increase characteristic value analysis to 20 clusters over all   
2. Add characteristic value analysis over both data sets - Currently only supports Modulus-Hardness data set
3. Add Scaled Difference plotting functinality for characteristic values page 
4. Add versus plotting functionality to plot two of the same characteristic value against each other on same graph
5. Refactor code to make code more efficient 
