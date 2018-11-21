#Install packages
list.of.packages <- c("e1071","keras","tensorflow","class","ggplot2","knitr","readr","caret","ggthemes","plotly","graphics","forcats")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

#Load packages
library(ggplot2)
library(knitr)
library(readr)
library(reshape)
library(gplots)
library(caret)
library(dplyr)
library(forcats)     # for "fct_reorder" function
library(graphics)     # dependancy fulfilment for plotly package
library(ggplot2)
library(ggthemes)
library(plotly)
library(MASS)
library(gains)
library(class)

#Load datatset
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
train <- read.csv("fashion-mnist_train.csv")
valid<- read.csv("fashion-mnist_test.csv")
dim(fashion.train)

#Labels
l.names <- c("T-shirt/Top",
                 "Trouser",
                 "Pullover",
                 "Dress",
                 "Coat",
                 "Sandal",
                 "Shirt",
                 "Sneaker",
                 "Bag",
                 "Ankle-Boot")

#Exploring no of observations per label name
label.factor <- as.factor(train$label)
levels(label.factor) <- l.names

# plot a random sample of product images
library(purrr)
xy_axis = data.frame(x = expand.grid(1:28,28:1)[,1],
                     y = expand.grid(1:28,28:1)[,2])

plot_theme = list(raster = geom_raster(hjust = 0, vjust = 0), 
                  gradient_fill = scale_fill_gradient(low = "white", high = "black", guide = FALSE), 
                  theme = theme(axis.title = element_blank(), panel.background = element_blank(),
                                panel.border = element_blank(),panel.grid.major = element_blank(),
                                panel.grid.minor = element_blank(), plot.background = element_blank(),
                                aspect.ratio = 1))

sample_plots = sample(1:nrow(train),35) %>% map(~ {
  plot_data = cbind(xy_axis, fill = as.data.frame(t(train[.x, -1]))[,1]) 
  ggplot(plot_data, aes(x, y, fill = fill)) + plot_theme
})

library(gridExtra)
do.call("grid.arrange", c(sample_plots, ncol = 7, nrow = 5))

#Making frequency chart
label_count_plot_train <- ggplot(data = train, aes(x = label, fill = as.factor(label))) + 
  geom_histogram(bins = 40) + 
  scale_x_continuous(breaks = seq(min(0), max(25), by = 1), na.value = TRUE) +
  scale_y_continuous(breaks = seq(min(0), max(10000), by = 200)) + 
  labs(title = "", x = "Labels", y = "Total Label Count", fill = "Label")+
  scale_fill_discrete(labels=c("0: T-shirt/Top",
                               "1: Trouser",
                               "2: Pullover",
                               "3: Dress",
                               "4: Coat",
                               "5: Sandal",
                               "6: Shirt",
                               "7: Sneaker",
                               "8: Bag",
                               "9: Ankle-Boot"))

ggplotly(label_count_plot_train, width = 700, height = 800)

sum(is.na(train))

#Data pre-processing Remove near zero variance variable
set.seed(1)
nzrv <- nearZeroVar(train[,-1], saveMetrics = T, freqCut = 300, uniqueCut = 1/4)
discard <- rownames(nzrv[nzrv$nzv,])
keep <- setdiff(names(train), discard)
trainnzv <- train[,keep]

cat(sum(nzrv$nzv), "near zero variance predictors have been removed,", "\n") 
cat(sum(nzrv$zeroVar), "of which were zero variance predictors.")


label <- as.factor(trainnzv$label)
trainnzv$label <- NULL
trainnzv <- trainnzv / 255

#Perform PCA to reduce dimensions
set.seed(1)
train.cov <- cov(trainnzv)
train.pc <- prcomp(train.cov)
options(max.print=1000000)
#summary(train.pc)
var.ex <- train.pc$sdev^2 / sum(train.pc$sdev^2)
var.cum <- cumsum(var.ex)

results <- data.frame(num <- 1:length(train.pc$sdev),
                      ex = var.ex,
                      cum = var.cum)
par(mar=c(1,1,1,1))
plot(results$num, results$cum, type = "b", xlim = c(0,20),
     main = "Variance Explained by Top 50 PC",
     xlab = "Number of Components", ylab = "Variance Explained")


#Replace train
train.score <- as.matrix(trainnzv) %*% train.pc$rotation[,1:50]
train.pca <- cbind(label, as.data.frame(train.score))

#Replace validation
label <- valid[,1]
validwolabels<- valid[,-1]
keep <- setdiff(names(validwolabels), discard)
validnzv <- validwolabels[,keep]
validnzv <- validnzv / 255
valid.score <- as.matrix(validnzv) %*% train.pc$rotation[,1:50]
valid.pca <- cbind(label, as.data.frame(valid.score))
valid.pca$label <- factor(valid.pca$label)

#Create dataframe to store model results
model.accuracytest<- setNames(data.frame(matrix(ncol = 3, nrow = 0)), 
                          c("model", "accuracy_test","SecsTaken_test"))

#########
#Random forest on PCA data
library(randomForest)

start.time <- Sys.time()
fmnist_rf=randomForest(label~. ,data = train.pca,ntree = 50)
t <- Sys.time() - start.time


predrf <- predict(fmnist_rf,valid.pca)
str(predrf)
str(valid.pca)
rf_cm <- confusionMatrix(predrf, 
                valid.pca$label,
                dnn = c("RF-Predicted", "Actual"))
model.accuracytest['rf',] <- c('rf', rf_cm$overall[1],as.numeric(t, units = "secs"))

######
#Model with SVM
library(e1071)
start.time <- Sys.time()
fmnist_svm <- svm(label ~ ., data=train.pca)
t<- Sys.time() - start.time

pred_svm <- predict(fmnist_svm, valid.pca)

svm_cm <- confusionMatrix(pred_svm, 
                valid.pca$label,
                dnn = c("SVM-Predicted", "Actual"))

model.accuracytest['svm',] <- c('svm', svm_cm$overall[1],as.numeric(t, units = "secs"))

###############
#Model with LDA
#Remarks : Very quick
library(MASS)


start.time <- Sys.time()
fmnist_lda <- lda(label~.,data = train.pca)
t<- Sys.time() - start.time

pred_lda <- predict(fmnist_lda, valid.pca)

lda_cm <- confusionMatrix(pred_lda$class, 
                valid.pca$label,
                dnn = c("LDA-Predicted", "Actual"))

model.accuracytest['lda',] <- c('lda', lda_cm$overall[1],as.numeric(t, units = "secs"))

###Neural Network--nnet
library(nnet)

n <- names(train.pca[,-1])
f <- as.formula(paste("label ~", paste(n[!n %in% "medv"], collapse = " + ")))
start.time <- Sys.time()
fmnist_nnet <- nnet(f,data = train.pca,
                    size=150,maxit=130,MaxNWts = 80000)
t<- Sys.time() - start.time

library(NeuralNetTools)

plotnet(fmnist_nnet,skip = TRUE)

pred_nnet <- predict(fmnist_nnet,valid.pca,type="class")
nnet_cm <- confusionMatrix(factor(pred_nnet), 
                valid.pca$label,
                dnn = c("nnet-Predicted", "Actual"))


model.accuracytest['nnet',] <- c('nnet', nnet_cm$overall[1],as.numeric(t, units = "secs"))

###k-nearest neighbours
##Remarks : Takes time to run
library(caret)
library('e1071')
trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(1)

start.time <- Sys.time()
fmnist_knn <- train(label ~., data = train.pca, method = "knn",
                    trControl=trctrl,preProcess = c("center", "scale"),
                    tuneLength = 10)
t<- Sys.time() - start.time



fmnist_knn
plot(fmnist_knn)

pred_knn <- predict(fmnist_knn, newdata = valid.pca)
pred_knn

knn_cm <- confusionMatrix(pred_knn, 
                valid.pca$label,
                dnn = c("knn-Predicted", "Actual"))

model.accuracytest['knn',] <- c('knn', knn_cm$overall[1],as.numeric(t, units = "secs"))


#########
####################VISUALIZE PREDICTIONS#####################3
#Visualize predictions with actual digits using SVM model
m1.preds <- predict(fmnist_svm, valid.pca, array.layout = "colmajor")
m1.preds[1:5]
label.name = m1.preds
categories = c("T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Boot")
levels(label.name) = categories # add category column (character)



##plot images with green as correct and red as wrong
validoriginal <- data.matrix(valid)
validrep<- validoriginal
validrep[,-1] <- validrep[,-1]/255

plotResults <- function(images, preds, name){
  op <- par(no.readonly=TRUE)
  x <- ceiling(sqrt(length(images)))
  par(mfrow=c(x,x), mar=c(.1,.1,.1,.1))
  
  for (i in images){
    m <- matrix(validrep[i,-1], nrow=28, byrow=TRUE)
    m <- apply(m, 2, rev)
    image(t(m), col=grey.colors(255), axes=FALSE)
    text(0.08,0.95,
         col=ifelse(preds[i]==validrep[i,1],"green","red"), 
         cex=1.0, name[i])
  }
  par(op)
}

plotResults(sample(1:length(m1.preds), 25, replace=F),
            m1.preds, label.name)

#create a loop function for random images every sec seconds
loop <- function(sec){
  i = 1
  while(TRUE){
    if (i %% 8 == 0){
      break           #A condition to break out of the loop
    }
    plotResults(sample(1:length(m1.preds), 25, replace=F),
                m1.preds, label.name)           #Run your code
    Sys.sleep(time = sec) #Time in seconds
    
    i = i + 1
  }
}

#run loop function
loop(sec=3)
