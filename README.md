# Exoplanet detection via DeepLearning


Scientists use data collected by space telescopes to find new information that allows us to learn more about the universe. In particular, the NASA Kepler space telescope has been collecting light from thousands of stars for many years to detect the presence of exoplanets. An exoplanet is a planet that orbits around a star, just like the Earth. However, these systems are hundreds or thousands of light years away from Earth, so it is essential to have tools that allow us to understand if the presence of exoplanets is likely.

The data collected by spatial telescopes is huge and the new Artificial Intelligence techniques allow an advanced data analysis and powerful predictive models. In this project we used a dataset from the kaggle site, which coming from the Mikulski Archive, a large archive of astronomical data.

First of all, I’m going to apply different feature engineering techniques to the dataset and then we'll present 2 models ( SVC and a FCN ) which will be used for classifying the brightness flux of stars. 
In particular, the models were used to analyze the curve light of over 5000 stars trying to reach a high accuracy, in the end we will show which are the best models for the classification of this problem.



## Dataset

The dataset that we've used is a kaggle dataset :

link dataset : https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data

This contains data of the luminosity flux of over 5000 stars. Precisely the dataset has dimensions (5087, 3198), which the first column is the target value.
An important thing to consider is the fact that the dataset is unbalanced. There are very few positive examples, less than 1% of the entire training set and the same in the test set. 



## Machine Learning Model

For this task ( version v1 ) we used a CNN and a SVC. CNN ( Convolution Nerual Newtork ) is probably one of the most famous nerual network in deep learning; 
SVC ( Support Vector Classification ) is a Machine Learing model that has good performance and could approach the results of deep neural network.


The problem of the exoplanet detection through the brighntess flux analysis is a problem called in Machine Learing TSC ( Time Series Classification ).
According some scientific papaer the best models for this type of problem are CNN/FCN, ResNet; In this project we considered CNN and we added the SVC model as well.

The architecure of the CNN was taken from a reasearch paper while the SVC model was built by us.




## Code and implementation


A Gauss filter (with sigma = 7) was applied to the signals to remove some error from the signals and then we applied the FFT to the data to obtain the frequency domain. Machine learning models worked with data in the frequency domain.

#### In the file : 

1) [models.py](https://github.com/senad96/exoplanet-detection-via-DeepLearning_v1/blob/main/models.py) you can find the CNN model and the SVC model.
2) [FDS_project.py](https://github.com/senad96/exoplanet-detection-via-DeepLearning_v1/blob/main/FDS_project.py) you can find the script that run all the operations. To run the code you have to run only this file.



## Testing and Performance evaluation

We've presented all the performance calssfication metrics: 

1) Accuracy
2) Recall
3) Precision
4) F1-score
5) Roc-curve



## Future work

Since this application is a standard in astrophysics, our models can be used on new and larger datasets by changing only the model input; 
( which are set to manage time series data long 3197 ).
For more details you can read the report file where all the work is explained better.









