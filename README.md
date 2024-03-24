# Network-Science---Link-Prediction-Challenge

This is a repository for the Kaggle competition within the "Machine Learning in Network Science" class at CentraleSupélec.
This is a challange about predicting the missing links in an actor co-occurance network. We reached an AUC of 85.3% ranked 1st in the the public score and 3rd in private competition among 40 teams.

Link to the competion: [https://www.kaggle.com/competitions/centralesupelec-mlns-2024/overview]

File Description:

- link_prediction.ipynb : This is the main notebook for link prediction challange.
- feature_selection.ipynb : This is where we run our feature selection methods, we iterate over 100 times with random sampling of the 19 features we come up with. We select the best features that give us a good balance between TPR and TNR
- pagerank_creation.ipynb : This notebook creates the rooted pagerank for every node pair and saves it in a json file. We call this json file when we are computing pagerank in our feature extractor.
- utils/dataprep.py : This file is the utility where all the functions we use are located.
- data : This folder contains the data from competition, rooted page rank json, our best submission file.
