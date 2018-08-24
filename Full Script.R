## ---------- Module 4 - Task 3 ---------- ##
### -------- Preprocess the Data -------- ###

pacman::p_load(readr,dplyr,anchors,mgcv)
setwd("~/Dropbox/Ubiqum Code Academy/Module 4/Task3")

trainingData <- read_csv("UJIndoorLoc/trainingData.csv")
valData <- read_csv("UJIndoorLoc/validationData.csv")

# WAP's Attributes that are never detected and have always value=100
#We find which attribute have 0 variance in teh ValidationData and we delete these attributes from the trainingData
#Then, from the rest of the attributes in the trainingData, we delete again the ones with 0 variance 
NoVarVal <- which(apply(valData, 2, var) == 0)
trainingData <- trainingData[,-NoVarVal]
valData <- valData[,-NoVarVal]
NoVarTr <- which(apply(trainingData, 2, var) == 0)
trainingData <- trainingData[,-NoVarTr]
valData <- valData[,-NoVarTr]

#Factorize the attributes nad delete the unimportant once
trainingData$FLOOR <- as.factor(trainingData$FLOOR)
trainingData$BUILDINGID <- as.factor(trainingData$BUILDINGID)
valData$FLOOR <- as.factor(valData$FLOOR)
valData$BUILDINGID <- as.factor(valData$BUILDINGID)
trainingData[,317:318]<-NULL
valData[,317:318]<-NULL

#Creating a DATETIME attribute
#trainingData$DATETIME <- as.POSIXct(trainingData$TIMESTAMP, origin="1970-01-01")

# Creating the values more vizilible
#I first replace values  =100 to -105
trainingData <-  replace.value(trainingData, c(1:312), from=100, to=as.integer(-105), verbose = FALSE)
valData <- replace.value(valData, c(1:312), from=100, to=as.integer(-105), verbose = FALSE)
# Then, I set the undetected WAPs as 0 and the highets signals as 105 by adding +105 in my matrix
trainingData[1:312] <- trainingData[,1:312]+105
valData[1:312] <- valData[,1:312]+105

#I will check if there is an error prediction with any detected WAP but gives predictions
## Training Data
tr<-c()
for (i in 1:nrow(trainingData)){
  if (sum(trainingData[i,1:312])==0){
    tr <- c(tr, i)
  } }
Checkrows<-trainingData[tr,]
trainingData <- trainingData[-c(tr),]
## Validation Data do not have rows with 0 sum

trainingData <- distinct(trainingData)

#IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII#
validationData <- valData

FullData <- rbind(trainingData,validationData)
FullData <- distinct(FullData)

#Normalization
NormFullData <- as.data.frame(t(apply(FullData[, 1:312], 1, function(x) (x - min(x))/(max(x)-min(x)))))
FullData[1:312] <- NormFullData[1:312]
NormValData <- as.data.frame(t(apply(validationData[, 1:312], 1, function(x) (x - min(x))/(max(x)-min(x)))))
validationData[1:312] <- NormValData[1:312]

FullData <- distinct(FullData) # 2 rows more are deleted #

remove(Checkrows,NormFullData,NormValData,i,NoVarTr,NoVarVal,tr)


## ---------- Module 4 - Task 3 ---------- ##
### ---------- Vizualization ------------ ###

pacman::p_load(DescTools,	scatterplot3d,plotly)

# Observation of the Data and the building shame
ggplot(trainingData, aes(x=LONGITUDE, y=LATITUDE, color=BUILDINGID)) +
  geom_point()
ggplot(validationData, aes(x=LONGITUDE, y=LATITUDE, color=BUILDINGID)) +
  geom_point()
ggplot(FullData, aes(x=LONGITUDE, y=LATITUDE, color=BUILDINGID)) +
  geom_point()

scatterplot3d(FullData$LONGITUDE, FullData$LATITUDE, FullData$FLOOR, pch = 20, angle = 120, 
              color = FullData$FLOOR, main = "Coordinates and Floors")


# Check distribution of signal strength
# traning data
x <- trainingData[,1:312]
x <- stack(x)

x <- x[-grep(0, x$values),]
hist(x$values, xlab = "WAP strength", main = "Distribution of WAPs signal stength (Training set)", 
     col = "red")


## ---------- Module 4 - Task 3 ---------- ##
### ------------- Modeling -------------- ###

pacman::p_load(caret,party,doMC,randomForest,mltools,scatterplot3d,plotly)

#Speed up the process using more PC cores
registerDoMC(cores = 6)

# Predictions - Predictions - Predictions - Predictions - Predictions - Predictions - Predictions #
#### BUILDINGID BUILDINGID BUILDINGID BUILDINGID BUILDINGID BUILDINGID BUILDINGID BUILDINGID #### 
# Creating data partition nad a 10-fold CV
# set.seed(31)
# inTraining <- createDataPartition(FullData$BUILDINGID, p=.75, list = FALSE)
# training <- FullData[inTraining,]
# testing <- FullData[-inTraining,]
# fitControl <- trainControl(method = "cv", number = 10)
# BIDModel <- train(BUILDINGID~.-LONGITUDE-LATITUDE-FLOOR,
#                    data = training, method = "svmRadial", trControl = fitControl,
#                    tuneLength = 5)
# BIDModel
# PrBUILDINGID <- predict(BIDModel,testing)
# postResample(PrBUILDINGID,testing$BUILDINGID)
# saveRDS(BIDModel,"BIDModel.rda")
BIDModel <- readRDS("BIDModel.rda")
# BUILDINGID Prediction in the validationData
PrBUILDINGID <- predict(BIDModel,validationData)
postResample(PrBUILDINGID,validationData$BUILDINGID)
BIDModelCM <- confusionMatrix(PrBUILDINGID, validationData$BUILDINGID)
BIDModelCM ### 100% Accuract ###
#### END BUILDINGID ####

#### LONGITUDE LONGITUDE LONGITUDE LONGITUDE LONGITUDE LONGITUDE LONGITUDE LONGITUDE LONGITUDE ####
# #Creating data partition nad a 10-fold CV
# inTraining <- createDataPartition(FullData$LONGITUDE, p=.80, list = FALSE)
# training <- FullData[inTraining,]
# testing <- FullData[-inTraining,]
# fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
# LONGModelRF <- randomForest(LONGITUDE~.-BUILDINGID-LATITUDE-FLOOR, data=training,importance=TRUE,
#                             proximity=TRUE, ntree=100, trainControl=fitControl)
# # LONGModel <- train(LONGITUDE~.-LATITUDE-BUILDINGID-FLOOR,
# #                    data = training, method = "knn", trControl = fitControl,
# #                    tuneLength = 5)
# # LONGModel
# PrLONGITUDE <- predict(LONGModelRF,testing)
# postResample(PrLONGITUDE,testing$LONGITUDE)
# saveRDS(LONGModelRF, "LONGModelRF.rda")
LONGModelRF <- readRDS("LONGModelRF.rda")
#LONGITUDE Prediction in the validationData
PrLONGITUDE <- predict(LONGModelRF, validationData)
postResample(PrLONGITUDE, validationData$LONGITUDE) # 5.1566559 0.9982149 #
LONGModel <- readRDS("LONGModel.rda")
PrLONGITUDE <- predict(LONGModel, validationData)
postResample(PrLONGITUDE, validationData$LONGITUDE) # 6.4662943 0.9971156 ###
#without preprocess because it gives me worse results
#### END LONGITUDE ####

#### LATITUDE LATITUDE LATITUDE LATITUDE LATITUDE LATITUDE LATITUDE LATITUDE LATITUDE LATITUDE ####
# Creating data partition nad a 10-fold CV
# set.seed(31)
# inTraining <- createDataPartition(FullData$LATITUDE, p=.80, list = FALSE)
# training <- FullData[inTraining,]
# testing <- FullData[-inTraining,]
# fitControl <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
# LATModelRF <- randomForest(LATITUDE~.-BUILDINGID-LONGITUDE-FLOOR, data=training,importance=TRUE,
#                             proximity=TRUE, ntree=100, trainControl=fitControl) 
# # fitControl <- trainControl(method = "cv", number = 10)
# # LATModel <- train(LATITUDE~.-BUILDINGID-FLOOR-LONGITUDE, 
# #                   data = training, method = "knn", trControl = fitControl, 
# #                   tuneLength = 5)
# # LATModel
# PrLATITUDE <- predict(LATModelRF,testing)
# postResample(PrLATITUDE,testing$LATITUDE)
# saveRDS(LATModelRF,"LATModelRF.rda")
LATModelRF <- readRDS("LATModelRF.rda")
# FLOOR Prediction in the validationData
PrLATITUDE <- predict(LATModelRF,validationData)
postResample(PrLATITUDE,validationData$LATITUDE) # 3.9002826 0.9970939 #
# with Knn I got 5.798817 0.993230 #
#### END LATITUDE ####

#### Re-ROTATE TEH COORDINATES ####
# # Checking if the rotation is ok
# CheckCoordinates <- Rotation(TrCoordinates, (-29)*pi/180)
# plot(CheckCoordinates)
#### ENDF Re-ROTATION ####



#### COORDINATED NORMALIZATION FOR FLOOR PREDICTION ####
NormCoords <- as.data.frame(apply(FullData[, 313:314], 2, function(x) (x - min(x))/(max(x)-min(x))))
FullData[317:318] <- NormCoords[1:2]
NormCoordsVal <- as.data.frame(apply(validationData[, 313:314], 2, function(x) (x - min(x))/(max(x)-min(x))))
validationData[317:318] <- NormCoordsVal[1:2]
#### END COORDINATED NORMALIZATION ####

#### FLOOR FLOOR FLOOR FLOOR FLOOR FLOOR FLOOR FLOOR FLOOR FLOOR FLOOR FLOOR FLOOR FLOOR FLOOR ####
# Creating data partition and a 10-fold CV
# inTraining <- createDataPartition(FullData$FLOOR, p=.80, list = FALSE)
# training <- FullData[inTraining,]
# testing <- FullData[-inTraining,]
# fitControl <- trainControl(method = "cv", number = 10)
# FLModel <- train(FLOOR~.-BUILDINGID-LONGITUDE-LATITUDE,
#                    data = training, method = "svmLinear", trControl = fitControl)
# FLModel
# PrFLOOR <- predict(FLModel,testing)
# postResample(PrFLOOR,testing$FLOOR)
# saveRDS(FLModel,"FLModel.rda")
FLModel <- readRDS("FLModel.rda")
# FLOOR Prediction in the validationData
PrFLOOR <- predict(FLModel,validationData)
postResample(PrFLOOR,validationData$FLOOR) #Normalizing the coords:Accuracy:0.9864986 Kappa :0.9810194 #
FLModelCM <- confusionMatrix(PrFLOOR, validationData$FLOOR)
FLModelCM
#### END FLOOR ####

FullData[317:318] <- NULL
validationData[317:318] <- NULL

#### RADIAL COORDINATES' ERROR IN PREDICTION ####
Predictions <- as.data.frame(cbind(PrLONGITUDE,PrLATITUDE,PrBUILDINGID,PrFLOOR))
Predictions$PrBUILDINGID<-as.factor(Predictions$PrBUILDINGID)
Predictions$PrFLOOR<-as.factor(Predictions$PrFLOOR)

plot_ly(Predictions, x = ~PrLATITUDE, y = ~PrLONGITUDE, z = ~PrFLOOR, color = ~PrFLOOR, sizes = 1/10000, colors = c('#BF382A', '#0C4B8E')) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Latitude'),
                      yaxis = list(title = 'Longitude'),
                      zaxis = list(title = 'Floor')))

ggplot() + geom_point(data=Predictions, aes(x=PrLONGITUDE, y=PrLATITUDE,col="Predictions")) + geom_point(data=validationData, aes(x=LONGITUDE, y=LATITUDE, col="RealData"))


scatterplot3d(Predictions$PrLONGITUDE, Predictions$PrLATITUDE, Predictions$PrFLOOR, pch = 20, angle = 120, 
              color = Predictions$PrFLOOR, main = "Predicted Coordinates and Floors")
ErrorCoord <- sqrt((Predictions$PrLONGITUDE - validationData$LONGITUDE)^2 + 
                     (Predictions$PrLATITUDE - validationData$LATITUDE)^2)
hist(ErrorCoord, freq = T, xlab = " Absolute error (m)", col = "green", main = "Error distance in meters", breaks = 50)
boxplot(ErrorCoord)
LargeErrorRowsNum <-  which(ErrorCoord>20)
LargeErrorRows <- Predictions[c(LargeErrorRowsNum),]
ErrorValrows <- validationData[c(LargeErrorRowsNum),]
ggplot() + geom_point(data=ErrorValrows, aes(x=LONGITUDE, y=LATITUDE, col="RealData"),size=5) + 
  geom_point(data=LargeErrorRows, aes(x=PrLONGITUDE, y=PrLATITUDE,col="Prediction"))+
  geom_point(data=validationData, aes(x=LONGITUDE, y=LATITUDE, col="ValData",alpha=.00000001))

ErrorLONG <- sqrt(((Predictions$PrLONGITUDE - validationData$LONGITUDE)^2))
hist(ErrorLONG, freq = T, xlab = " Absolute error (m)", col = "red", main = "Error distance in meters")
boxplot(ErrorLONG)
Error1 <- which(ErrorLONG>30)

ErrorLAT <- sqrt(((Predictions$PrLATITUDE - validationData$LATITUDE)^2))
hist(ErrorLAT, freq = T, xlab = " Absolute error (m)", col = "red", main = "Error distance in meters")
boxplot(ErrorLAT)
Error2 <- which(ErrorLAT>30)

Errors<- rbind(validationData[c(Error1),],validationData[c(Error2),])

scatterplot3d(LargeErrorRows$LONGITUDE, LargeErrorRows$LATITUDE, LargeErrorRows$FLOOR, pch = 20, angle = 120, 
              color = LargeErrorRows$FLOOR, main = "Error Coordinates and Floors")
ggplot(LargeErrorRows, aes(x=LONGITUDE, y=LATITUDE, color=BUILDINGID)) + geom_point()
#### End coordinates' error ####

#### FLOOR PREDICTION ERRORS ####
FLOORErros <- validationData[which(validationData$FLOOR != PrFLOOR),]
#### ####

remove(NormCoords,NormCoordsVal)