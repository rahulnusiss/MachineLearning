#==================Multi layer Perceptron BP Start============================================

install.packages('nnet')
install.packages('NeuralNetTools')

library(nnet)
library(NeuralNetTools)

data <- read.csv("E:\\Study\\Computational Intelligence\\Diabetes.csv",head=TRUE,sep=",")
summary(data)

# shuffle
x <- data[sample(1:nrow(data)),]

#traindata = sample(1:768,576)
traindata = 1:576
#valdata = setdiff(1:768, traindata)
valdata = 577:768
#data[traindata,]

train <- data[1:576,]
test <- data[577:768,]

#summary(data)
#summary(traindata)
#summary(valdata)

# -9 because we dont want to use diabetic classification as our input
ideal <- class.ind(data$diabeticclass)
timeBP1 = Sys.time() #time calculation
dataANN = nnet(data[traindata,-9], ideal[traindata,], size=10, softmax=TRUE, maxit=1000)
timeBP2 = Sys.time()
pred = predict(dataANN, data[valdata,-9], type="class")
timeBP3 = Sys.time() 
timediffBPModel = timeBP2-timeBP1
timediffBPPredict = timeBP3-timeBP2
table(true=test$diabeticclass, predicted=pred)

plotnet(dataANN, pos_col = "green", neg_col = "red")
#summary(data)

resultBP = cbind(test, data.frame(pred))
resultBP.correct=nrow(result[resultBP$pred == resultBP$diabeticclass,])
resultBP.size=nrow(resultBP)
cat("No of test cases = ",resultBP.size,"\n")
cat("Correct predictions = ", resultBP.correct ,"\n")
cat("Accuracy = ", resultBP.correct / 192 * 100 ,"\n")

#==================Multi layer Perceptron BP End============================================

#=================Probabilistic Neural Network(PNN)============
install.packages('pnn')
library(pnn)

data <- read.csv("E:\\Study\\Computational Intelligence\\Diabetes.csv",head=TRUE,sep=",")

size=nrow(data)

length=ncol(data)

index <- 1:size

#training <- data[positions,]
#testing <- data[-positions,1:length-1]
training <- data[1:576,]
testing <- data[577:768,1:length-1]
resultPNN= data[577:768,]
resultPNN$actual = resultPNN[,length]
resultPNN$predict = -1

timePNN1 = Sys.time()
pnn1 <- learn(training, pnn1,category.column = 9)
pnn1 <- smooth(pnn1, sigma = 0.95)
timePNN2 = Sys.time()
for(i in 1:nrow(testing))
{	
  vec <- as.matrix(testing[i,])
  res <- guess(pnn1, vec)
  
  if(is.na(res))
  {
    cat("Entry ",i," Generated NaN result!\n")
  }
  else
  {
    resultPNN$predict[i] <- res$category
  }
}

timePNN3 = Sys.time()
timediffPNNModel = timePNN2-timePNN1
timediffPNNPredict = timePNN3-timePNN2

resultPNN.size = nrow(resultPNN)
table(true=resultPNN$actual, predicted=resultPNN$predict)
resultPNN.correct = nrow(resultPNN[resultPNN$predict == resultPNN$actual,])
cat("No of test cases = ",resultPNN.size,"\n")
cat("Correct predictions = ", resultPNN.correct ,"\n")
cat("Accuracy = ", resultPNN.correct / resultPNN.size * 100 ,"\n") 

write.csv(resultPNN, file="E:\\Study\\Computational Intelligence\\resultDiabetesPNN.csv")

#==================PNN End=============================

#==========Radial Basis Function(RBF)====================
install.packages("RSNNS")

library("Rcpp")
library("RSNNS")

data = read.csv("E:\\Study\\Computational Intelligence\\Diabetes.csv", header=TRUE)


X = data[1:576,1:8]
Y = data[1:576,9]

X.out = data[577:768,1:8]
Y.out = data[577:768,9]
N.test = 192

ideal <- class.ind(Y)
outputs <- as.matrix(Y)

timeRBF1 = Sys.time() 

rbfn.model <- RSNNS::rbf(as.matrix(X), 
                         ideal, 
                         size=40,    # number of centers, ie, number of neurons in hidden layer
                         maxit=1000, # max number of iterations to learn
                         linOut=TRUE, # TRUE = linear activation function (otherwise logistic)
                         initFunc = "RBF_Weights",
                         learnFunc = "RadialBasisLearning",
                         initFuncParams=c(0, 1, 0, 0.01, 0.01),
                         learnFuncParams=c(1e-8, 0, 1e-8, 0.1, 0.8))

timeRBF2 = Sys.time()

predicted = predict(rbfn.model, X.out)
rbf.network.pred <- sign( predicted ) # apply sign, since this is a classification eg

timeRBF3 = Sys.time()               #Calculate time difference
timediffRBFModel = timeRBF2-timeRBF1
timediffRBFPredict = timeRBF3-timeRBF2

resultRBF = -1
for(i in 1:192)
{
  resultRBF[i] = ifelse(predicted[i,1] > predicted[i,2], 0,1)
}

binary.accuracy <- sum(resultRBF == Y.out)/N.test
binary.accuracy 

#==========Radial Basis Function(RBF) End==============================


#========================= Ensemble MLP_BP, PNN, RBF => Voting ==========================

isDiabetic = -1 # For ensemble predict result
timeVote1 = Sys.time()
for(i in 1:192)
{	
  resPNN = as.numeric(resultPNN$predict[i])
  
  if(resultBP$pred[i] == resPNN && resultBP$pred[i] == resultRBF[i])
  {
    #isDiabetic[i] = more than 1 in R. To avoid this following way implemented
    isDiabetic[i] = ifelse(resultBP$pred[i]==0,0,1)
  }
  
  if(resultBP$pred[i] == resPNN && resultRBF[i] != resultBP$pred[i])
  {
    isDiabetic[i] = ifelse(resultBP$pred[i]==0,0,1)
  }
  else if(resultRBF[i] == resPNN && resultRBF[i] != resultBP$pred[i])
  {
    isDiabetic[i] = ifelse(resultRBF[i]==0,0,1)
  }
  else
  {
    isDiabetic[i] = ifelse(resultBP$pred[i]==0,0,1)
  }
}
timeVote2 = Sys.time()
timediffEnsembleVote = timeVote2-timeVote1
resultEnsemble = isDiabetic
resultEnsemble.correct=nrow(result[isDiabetic == resultPNN$actual,])
resultEnsemble.size = 192

table(true=resultPNN$actual, VoteEnsemblePredict=isDiabetic)
cat("No of test cases = ",resultEnsemble.size,"\n")
cat("Correct predictions By Ensemble= ", resultEnsemble.correct ,"\n")
cat("Ensemble Accuracy = ", resultEnsemble.correct / resultEnsemble.size * 100 ,"\n")

