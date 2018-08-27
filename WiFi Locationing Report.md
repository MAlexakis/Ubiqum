# Indoor-Wifi-Locationing
This project is to predict users' location (Building, Floor and Coordinates) at Jaume I University from WAPs signal information using classification and regression machine learning algorithms

## About the datasets
Two datasets, training and validation, contain information about 520 WAPs signal strengths, three buildings and their floors, coordinates of users who logged in, space where the users logged in and the relative position (inside or outside the room), user ID, phone ID, and timestamp. 
The data was collected at Jaume I University.
More information about the dataset can be found on the source link provided below:
http://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc

## Objective
The objective is to build models that predict the location (building, floor and coordinates) from the WAPs signal strengths of a user who connects to the internet at Jaume I University. 

## Pre-Process
After intalling the Datasets I deleted the no-variance attributes from every dataset. After I deleted attributes that are not usefull since they do not provide valuable information about the predictions and I factorised BUILDINGID and FLOOR fro both Datasets.

In addition, I replaced all the values of 100 to -105 and then I added to the WAPS attributes +105 so I have values from zero (non detected waps) to 105 (detected waps).

After, I created a look that checks rows without any wap to be detected (zero variance rows) and later I delete duplicates. I compined the training and validation data so that to create a stronger model. 

In contineous, I normalize the waps attributes since that gives slightly better results in the predictions and I deleted again duplicates after the compination.

## Visualization
### Observation of the Data and the building shape of validation and training sets

###### 1. Training set graph
![1. Training set](https://user-images.githubusercontent.com/42608658/44660289-7a691480-aa07-11e8-841d-8feec2cc13d6.png)


###### 2. Validation set graph
![2. Validation set](https://user-images.githubusercontent.com/42608658/44660290-7a691480-aa07-11e8-9be5-4d8f7811be0c.png)


###### 3. A 3D Graph that shows the distribution of the floors
![3. 3D Graph that shows the distribution of the floors](https://user-images.githubusercontent.com/42608658/44660291-7a691480-aa07-11e8-990d-8a8feafc960a.png)


###### 4. Satelite capture of Jaume I University
![Satelite Capture](https://user-images.githubusercontent.com/42608658/44661384-cf5a5a00-aa0a-11e8-8f2f-655d284eb833.jpg)


###### 5. WAPs signal strength distribution
![WAPs signal strength distribution](https://user-images.githubusercontent.com/42608658/44661200-36c3da00-aa0a-11e8-8ea9-2984fdaf8e7e.png)


## Machine Learning Application and Results

After the a lot of improvements of the pre-process and normalization I ended up using the following algorithms for the prediction of every dependent variable.

### 1st Step. Building prediction (Classification with SVM algorith with Radial Kernel)
For the Building ID prediction I used SVM Radial Kernel algorith and my accuracy was 100% as it is shown below.

_**· Accuracy = 1**_

_**· Kappa = 1**_

_**· Confusion Matrix:**_

<img width="201" alt="screen shot 2018-08-27 at 15 22 55" src="https://user-images.githubusercontent.com/42608658/44662142-219c7a80-aa0d-11e8-9fc4-f8e3a4117692.png">
         
### 2st Step. Longitude prediction (Regression with Random Forest algorithm)
For the Longitude prediction I used the RF package and the rf() function fro prediction. That happened becausewith the train function parameter it was running out of time. So, the performance of the algorith is shown below.

_**· RMSE: 5.1566559**_

_**· Rsquared: 0.9982149**_

_**· MAE: 3.1034450**_

### 3st Step. Longitude prediction (Regression with Random Forest algorithm)
Equally, or the Latitude prediction I used the RF package and the rf() function fro prediction. The performance of the algorith is shown below.

_**· RMSE: 4.9970688**_

_**· Rsquared: 0.9952367**_

_**· MAE: 2.7341774**_

### 4st Step. Floor prediction (Classification with SVM Linear algorithm)
For the Floor prediction I use the Longitude and Latitude attributes (in a real case after the prediction of them two), but I had slightly better results by normalizing the coordinates and I created 2 new attributes with the normalized values (deleting them 2 new attributes after the prediction process, so I do not touch the original coordinates). Thus, the performans of the algorith is shown bellow.

_**· Accuracy = 0.9828983**_

_**· Kappa = 0.9759607**_

_**· Confusion Matrix:**_

<img width="246" alt="screen shot 2018-08-27 at 15 50 31" src="https://user-images.githubusercontent.com/42608658/44663482-f87de900-aa10-11e8-9dd9-c7e58b23558c.png">

**If you woulg like to see a virtual 3D graph of the final prediction, follow the link below:**
http://localhost:21116/session/viewhtml3e8044b5d5e6/index.html

## Errors' Vizualization

###### 1. Radial Error distribution
![radial error distribution](https://user-images.githubusercontent.com/42608658/44664278-11879980-aa13-11e8-838e-7eec7ba073ba.png)

###### 2. Comparison Between Large errors and real values of radial coordinates
The chart bellow shows the real values in blue and it highlits with big red dots the observations that give large wrong predictions. The small red dots are the wrong predictions.
![comparison between validation data and predictions](https://user-images.githubusercontent.com/42608658/44664397-6b885f00-aa13-11e8-82ef-a88fd0c07e70.png)
