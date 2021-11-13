# Transit Exoplanet Detection via Machine Learning


Scientists use data collected by space telescopes to find new information that allows us to learn more about the universe. The NASA Kepler space telescope has been collecting light from thousands of stars for many years to detect the presence of exoplanets.

An exoplanet is a planet that orbits a star, just like the Earth; however these systems are hundreds or thousands of light years away from Earth, so it is essential to have tools that can assist scientists in understanding whether a given star is likely to have exoplanets. The data collected by space telescopes is huge and new artificial intelligence techniques enable advanced data analysis and powerful predictive models.


In this project we used a dataset from the kaggle site, coming from the Mikulski Archive, a large archive of astronomical data for classifying the light curve of the stars to check the presence of the exoplanets. First of all, I'm going to apply different feature engineering techniques to the dataset and then we will present a neural network ( CNN ), which is a strong model in Deep Learning for time series classification ( TSC ), and a SVC model which will be used for classifying the brightness flux of stars. We will compare these 2 models in this task and I’ll show how a deep learning approach in this problem achieves a great performance. Since the measure of brightness is a standard in these applications, these predictive models can be useful for future works with other and new larger dataset.




## Dataset

The dataset that we've used is a kaggle dataset :

link dataset : https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data

This contains data of the luminosity flux of over 5000 stars. Precisely the dataset has dimensions (5087, 3198), which the first column is the target value.
An important thing to consider is the fact that the dataset is unbalanced. There are very few positive examples, less than 1% of the entire training set and the same in the test set. Here below we can see an example of signals and the respective smoothed version.

![ singal1 ](https://github.com/senad96/exoplanet-detection-via-DeepLearning_v1/blob/main/Images/signal1.png?raw=true)


![ singal2 ](https://github.com/senad96/exoplanet-detection-via-DeepLearning_v1/blob/main/Images/singall2_gauss.png?raw=true)


## Machine Learning Model

For this task, we used a CNN and a SVC. CNN ( Convolution Neural Newtork ) is probably one of the most famous nerual network in deep learning; 
SVC ( Support Vector Classifier ) is a Machine Learing model that has good performance and could approach the results of deep neural network.


The problem of the exoplanet detection through the brighntess flux analysis is a TSC ( Time Series Classification ) probelm
According some scientific paper the best models for this kind of problem are CNN/FCN, ResNet; In this project we considered a standard CNN ( with our own architecture ).

The architecure of the CNN was taken from a reasearch paper while the SVC model was built by us.

Above you can find a Report when you can read all the details and the results, even the References.




## Code and implementation


A gaussian filter (with sigma = 5) was applied to the signals to remove some error from the signals and then we applied the FFT to the data to obtain the frequency domain. Machine learning models worked with data in the frequency domain.

#### In the file : 

1) [models.py](https://github.com/senad96/exoplanet-detection-via-DeepLearning_v1/blob/main/models.py) you can find the CNN model and the SVC model.
2) [project_main.py](https://github.com/senad96/exoplanet-detection-via-DeepLearning_v1/blob/main/project_main.py) you can find the script that run all the operations. To run the code you have to run only this file.



## Testing and Performance evaluation

We've presented all the performance calssfication metrics: 

1) Accuracy
2) Recall
3) Precision
4) F1-score
5) Confusion Matrix

## Result

The CNN has achieved incredible results. With all the problems written above with a sufficient number of epochs (according to our tests they must be greater than 15) the model is able to detect 5 exoplanet in the test set ( 100 % accuracy ).
However the SVC fails the recognition of 4 exoplanets ( 1 is predicted correctly ).




## Future work

In the field of applied astrophysics the transit detection of exoplanets is an important technique for the discovery of new planets.
It would be very interesting to test our models ( in particular CNN ) in new even larger datasets. Recall that in the MAST archive ( Mikulski Archive for Space Telescopes ) it is possible to obtain the light curves of thousands of other stars collected by the kepler telescope. However these data are not in a form directly usable by a machine learning model and therefore would require to be downloaded and processed to extract the light curves.
In this project due to a limited time it was not possible to test the models with new datasets. In case someone would like to do this the code has been constructed to be able to change a number of reduced variables to adapt the models to new datasets. 

#### In particular the things to change in the code would be:


1) Put the new datasets in the variable "data_train", "data_test"
2) Check the target label in the training set and test set is the first column
3) The CNN input number. In our case the network accepts 3197 long sequences. In other cases it could be less or more.
   NOTE : In the scenario where the signals are of different lengths, it is possible to make them all of the same length by choosing a fixed number of values     
          after he application of FFT ( for example frequencies with greater amplitude).
5) Eventually change the hyper-parameters ( sigma of gaussian filter, layers of CNN etc. ) 



#### In the report above, you can find much more details about implementation.






