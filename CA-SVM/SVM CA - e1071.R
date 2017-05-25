
# un-comment if not installed
#install.packages("ISLR")
#

library(ISLR)
library(e1071)
setwd("D:/Users/Evon/Documents/NUS/MTech/AY2016_2017 Sem 2/Basic Electives/Computer Intelligence/CA-Reading")
#
# step 0: load the data
#         perform some data exploration
#
data(Default)
str(Default)
head(Default)
names(Default)

# 
# step 1: partition training and test data 
#         training (80%) 
#         testing  (20%)
#
set.seed(3)
total <- nrow(Default)
train <- sample(1:total, 8000)
trainData <- Default[train,]
testData <- Default[-train,]
trainData$stud <- as.numeric(trainData$student)
testData$stud <- as.numeric(testData$student)

#
# step 2: training and testing model(s) with different kernel function, gamma and cost 
#         (experiments with various combinations)
# 
# Model 1.1: svm with kernel function: radial basis
#            predictors: income+balance
#
tune1.out <- tune.svm(default~income+balance, data=trainData, kernel="radial")
png("SVM-Model1-radial-tune.png")
plot(tune1.out$best.model, trainData, income~balance, svSymbol=17, dataSymbol=1)
dev.off()
bestmod1 <- tune1.out$best.model
print(summary(bestmod1))

# testing
predictModel1 <- predict(tune1.out$best.model, testData)
summary(predictModel1)
resultModel1 <- table(prediction = predictModel1, truth = testData$default)

# Model 1.2: svm with kernel function: radial basis
#            predictors: student+income
#
tune12.out <- tune.svm(default~stud+income, data=trainData, kernel="radial")
png("SVM-Model12-radial-tune.png")
plot(tune12.out$best.model, trainData, stud~income, svSymbol=17, dataSymbol=1, lwd=1)
dev.off()
bestmod12 <- tune12.out$best.model
print(summary(bestmod12))

# testing
predictModel12 <- predict(tune12.out$best.model, testData)
summary(predictModel12)
resultModel12 <- table(prediction = predictModel12, truth = testData$default)

print("Result: SVM with kernel function rbf")
print("Predictors: income-balance")
print(resultModel1)
print((resultModel1[1,1] + resultModel1[2,2]) / nrow(testData) *100)
print("Predictors: student-income")
print(resultModel12)
print((resultModel12[1,1] + resultModel12[2,2]) / nrow(testData) *100)


#
# Model 2.1: svm with kernel function: polynomial
#            predictors: income+balance
#
tune2.out <- tune.svm(default~income+balance, data=trainData, kernel="polynomial")
png("SVM-Model2-poly-tune.png")
plot(tune2.out$best.model, trainData, income~balance, svSymbol=17, dataSymbol=1)
dev.off()
bestmod2 <- tune2.out$best.model
print(summary(bestmod2))

# testing
predictModel2 <- predict(tune2.out$best.model, testData)
summary(predictModel2)
resultModel2 <- table(prediction = predictModel2, truth = testData$default)

# Model 2.1: svm with kernel function: polynomial
#            predictors: student+income
#
tune21.out <- tune.svm(default~stud+income, data=trainData, kernel="polynomial")
png("SVM-Model21-poly-tune.png")
plot(tune21.out$best.model, trainData, stud~income, svSymbol=17, dataSymbol=1)
dev.off()
bestmod21 <- tune21.out$best.model
print(summary(bestmod21))

# testing
predictModel21 <- predict(tune21.out$best.model, testData)
summary(predictModel21)
resultModel21 <- table(prediction = predictModel21, truth = testData$default)

print("Result: with kernel function polynomial")
print("Predictors: income-balance")
print(resultModel2)
print((resultModel2[1,1] + resultModel1[2,2]) / nrow(testData) *100)
print("Predictors: student-income")
print(resultModel21)
print((resultModel21[1,1] + resultModel21[2,2]) / nrow(testData) *100)


#
# Model3.1: svm with kernel function: sigmoid
#            predictors: income+balance
#
tune3.out <- tune.svm(default~income+balance, data=trainData, kernel="sigmoid")
png("SVM-Model3-sigmoid-tune.png")
plot(tune3.out$best.model, trainData, income~balance, svSymbol=17, dataSymbol=1)
dev.off()
bestmod3 <- tune3.out$best.model
print(summary(bestmod3))

# testing
predictModel3 <- predict(tune3.out$best.model, testData)
summary(predictModel3)
resultModel3 <- table(prediction = predictModel3, truth = testData$default)

# Model3.1: svm with kernel function: sigmoid
#            predictors: student+income
#
tune31.out <- tune.svm(default~stud+income, data=trainData, kernel="sigmoid")
png("SVM-Model31-sigmoid-tune.png")
plot(tune31.out$best.model, trainData, stud~balance, svSymbol=17, dataSymbol=1)
dev.off()
bestmod31 <- tune31.out$best.model
print(summary(bestmod31))

# testing
predictModel31 <- predict(tune31.out$best.model, testData)
summary(predictModel31)
resultModel31 <- table(prediction = predictModel31, truth = testData$default)

print("Result: with kernel function sigmoid")
print("Predictors: income-balance")
print(resultModel3)
print((resultModel3[1,1] + resultModel1[2,2]) / nrow(testData) *100)
print("Predictors: student-income")
print(resultModel31)
print((resultModel31[1,1] + resultModel31[2,2]) / nrow(testData) *100)

