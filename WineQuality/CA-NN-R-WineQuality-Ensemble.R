
#==================Multi layer Perceptron BP Start============================================

install.packages('nnet')
install.packages('NeuralNetTools')

library(nnet)
library(NeuralNetTools)

data <- read.csv("E:\\Study\\Computational Intelligence\\winequality-white.csv",head=TRUE,sep=",")
summary(data)

# shuffle
x <- data[sample(1:nrow(data)),]

traindata = 1:3673

valdata = 3674:4898


train <- data[1:3673,]
test <- data[3674:4898,]

#summary(data)
#summary(traindata)
#summary(valdata)

# -12 because we dont want to use quality classification as our input
ideal <- class.ind(data$quality)
timeBP1 = Sys.time() #time calculation
dataANN = nnet(data[traindata,-12], ideal[traindata,], size=11, softmax=TRUE, maxit=1000)
timeBP2 = Sys.time()
pred = predict(dataANN, data[valdata,-12], type="class")
predictMLBP = pred
timeBP3 = Sys.time() 
timediffBPModel = timeBP2-timeBP1
timediffBPPredict = timeBP3-timeBP2
table(true=test$quality, predicted=pred)

plotnet(dataANN, pos_col = "green", neg_col = "red")
#summary(data)

resultBP = cbind(test, data.frame(pred))
resultBP.correct=nrow(result[resultBP$pred == resultBP$quality,])
resultBP.size=nrow(resultBP)
cat("No of test cases = ",resultBP.size,"\n")
cat("Correct predictions = ", resultBP.correct ,"\n")
cat("Accuracy = ", resultBP.correct / 1225 * 100 ,"\n")

#==================Multi layer Perceptron BP End============================================

#=================General Regression Neural Network(GRNN)============
install.packages('grnn')
library(grnn)

data <- read.csv("E:\\Study\\Computational Intelligence\\winequality-white.csv",head=TRUE,sep=",")

size=nrow(data)

length=ncol(data)

index <- 1:size

#training <- data[positions,]
#testing <- data[-positions,1:length-1]
training <- data[1:3673,]
testing <- data[3674:4898,1:length-1]
resultGRNN= data[3674:4898,]
resultGRNN$actual = resultGRNN[,length]
resultGRNN$predict = -1

timeGRNN1 = Sys.time()
grnn1 <- learn(training, variable.column=length)
grnn1 <- smooth(grnn1, sigma = 0.5)
timeGRNN2 = Sys.time()
for(i in 1:nrow(testing))
{	
  vec <- as.matrix(testing[i,])
  res <- guess(grnn1, vec)
  
  if(is.nan(res))
  {
    cat("Entry ",i," Generated NaN result!\n")
  }
  else
  {
    resultGRNN$predict[i] <- res
  }
}

timeGRNN3 = Sys.time()
timediffGRNNModel = timeGRNN2-timeGRNN1
timediffGRNNPredict = timeGRNN3-timeGRNN2

resultGRNN$predictclass=round(resultGRNN$predict)
predictGRNN = resultGRNN$predictclass
resultGRNN.size = nrow(resultGRNN)
table(true=resultGRNN$actual, predicted=resultGRNN$predictclass)
resultGRNN.correct = nrow(resultGRNN[resultGRNN$predictclass == resultGRNN$actual,])
cat("No of test cases = ",resultGRNN.size,"\n")
cat("Correct predictions = ", resultGRNN.correct ,"\n")
cat("Accuracy = ", resultGRNN.correct / resultGRNN.size * 100 ,"\n") 

write.csv(resultGRNN, file="E:\\Study\\Computational Intelligence\\resultWineQualityGRNN.csv")

#==================GRNN End=============================

#==========Radial Basis Function(RBF)====================
install.packages("RSNNS")

library("Rcpp")
library("RSNNS")

data = read.csv("E:\\Study\\Computational Intelligence\\winequality-white.csv", header=TRUE)


X = data[1:3673,-12]
Y = data[1:3673,12]

X.out = data[3674:4898,1:11]
Y.out = data[3674:4898,12]
N.test = 1225

resultRBF = Y.out
resultRBF = -1

ideal <- class.ind(Y)
outputs <- as.matrix(Y)

timeRBF1 = Sys.time() 

#as.matrix(X)
#initFuncParams=c(0.3, 0.3, 0.3, 0.3, 0.3,0.3,0.3),
#learnFuncParams=c(1e-05, 0, 1e-03,0.09, 0.1, 0.4, 0.8))
rbfn.model <- RSNNS::rbf(as.matrix(X), 
                         ideal, 
                         size=7,    # number of centers, ie, number of neurons in hidden layer
                         maxit=1000, # max number of iterations to learn
                         linOut=FALSE, # TRUE = linear activation function (otherwise logistic)
                         initFunc = "RBF_Weights",
                         learnFunc = "RadialBasisLearning",
                         initFuncParams=c(0.3, 0.3, 0.3,0.3,0.3,0.3,0.3),
                         learnFuncParams=c(1e-05, 0, 1e-03, 0.09, 0.1, 0.4, 0.8))

timeRBF2 = Sys.time()

predicted = predict(rbfn.model, X.out)

timeRBF3 = Sys.time()               #Calculate time difference
timediffRBFModel = timeRBF2-timeRBF1
timediffRBFPredict = timeRBF3-timeRBF2

# Class with highest probability is chosen to be the output result for each row.

max = -10
for(i in 1:1225)
{
  resultRBF[i] = which.max(predicted[i,]) + 2
}
predictRBF = resultRBF
#target,predicted
confusionMatrix(Y.out, resultRBF)

binary.accuracy <- sum(resultRBF == Y.out)/N.test
binary.accuracy 

#==========Radial Basis Function(RBF) End==============================


#========================= Ensemble MLP_BP, PNN, RBF => Weighted average ==========================

#weighted average 60% Multi layer perceptron, 20% GRNN, 20% RBF
actualresult_avg = Y.out
winequality_avg = 1:1225 # For ensemble predict result
timeaveraging1 = Sys.time()
for(i in 1:1225)
{	
  winequality_avg[i] = round(0.6*as.numeric(predictMLBP[i]) + 0.2*predictGRNN[i] + 0.2*predictRBF[i])
}
timeaveraging2 = Sys.time()
timediffEnsembleAveraging = timeaveraging2-timeaveraging1
resultEnsemble_avg = winequality_avg
resultEnsemble_avg.correct=nrow(result[winequality_avg == actualresult_avg,])
resultEnsemble_avg.size = 1225

table(true=actualresult_avg, WineEnsemblePredict_avg=winequality_avg)
cat("No of test cases = ",resultEnsemble_avg.size,"\n")
cat("Correct predictions By Ensemble= ", resultEnsemble_avg.correct ,"\n")
cat("Ensemble Accuracy Weighted Average= ", resultEnsemble_avg.correct / resultEnsemble_avg.size * 100 ,"\n")

