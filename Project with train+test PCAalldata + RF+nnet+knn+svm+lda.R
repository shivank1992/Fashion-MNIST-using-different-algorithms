###Install packages
list.of.packages <- c("doParallel","e1071","keras","tensorflow","class","ggplot2","knitr","readr","caret","ggthemes","plotly","graphics","forcats")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

###Load packages
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

#Enable Parallel processing
registerDoSEQ()
library(doParallel)
cl <- parallel::makeCluster(detectCores(logical=FALSE), type='PSOCK')
doParallel::registerDoParallel(cl)

###Load datatset
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
train <- read.csv("fashion-mnist_train.csv")
valid<- read.csv("fashion-mnist_test.csv")
dim(fashion.train)

###Labels
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

###Exploring no of observations per label name
label.factor <- as.factor(train$label)
levels(label.factor) <- l.names

###Plot a random sample of product images
library(purrr)
xy_axis = data.frame(x = expand.grid(1:28,28:1)[,1],
                     y = expand.grid(1:28,28:1)[,2])

plot_theme = list(raster = geom_raster(hjust = 0, vjust = 0), 
                  gradient_fill = scale_fill_gradient(low = "white", high = "black", guide = FALSE), 
                  theme = theme(axis.title = element_blank(), panel.background = element_blank(),
                                panel.border = element_blank(),panel.grid.major = element_blank(),
                                panel.grid.minor = element_blank(), plot.background = element_blank(),
                                aspect.ratio = 1))

sample_plots = sample(1:nrow(train),25) %>% map(~ {
  plot_data = cbind(xy_axis, fill = as.data.frame(t(train[.x, -1]))[,1]) 
  ggplot(plot_data, aes(x, y, fill = fill)) + plot_theme
})

library(gridExtra)
#png('Data Explored in Images.png')
do.call("grid.arrange", c(sample_plots, ncol = 5, nrow = 5))
dev.off()

###Making frequency chart
label_count_plot_train <- ggplot(data = train, aes(x = label, fill = as.factor(label))) + 
  geom_histogram(bins = 40) + 
  scale_x_continuous(breaks = seq(min(0), max(25), by = 1), na.value = TRUE) +
  scale_y_continuous(breaks = seq(min(0), max(10000), by = 1000)) + 
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

#png('Data Explored in Categories.png')
label_count_plot_train
dev.off()

ggplotly(label_count_plot_train, width = 700, height = 800)
dev.off()
#sum(is.na(train))

label_means <- train %>%
  group_by(label)%>%
  dplyr::summarize_all(mean)%>%
  rowMeans()

group_means <- train %>%
  group_by(label)%>%
  dplyr::summarize_all(mean)%>%
  rowMeans()

labelmeans<- data.frame(l.names,label_means)
labelmeans[order(labelmeans$label_means,decreasing = TRUE),]

###Data pre-processing Remove near zero variance variable
set.seed(1)
nzrv <- nearZeroVar(train[,-1], saveMetrics = T, freqCut = 300, uniqueCut = 1/4)
discard <- rownames(nzrv[nzrv$nzv,])
keep <- setdiff(names(train), discard)
trainnzv <- train[,keep]

discard
cat(sum(nzrv$nzv), "near zero variance predictors have been removed,", "\n") 
cat(sum(nzrv$zeroVar), "of which were zero variance predictors.")


label <- as.factor(trainnzv$label)
trainnzv$label <- NULL
trainnzv <- trainnzv / 255

###Perform PCA to reduce dimensions
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

plot(train.pc)
screeplot(train.pc, type="line", main="PCA Scree plot")
biplot(train.pc)



#par(mar=c(1,1,1,1))

#png('PCA Plot.png')
plot(results$num, results$cum, type = "b", xlim = c(0,20),
     main = "99.9% Variance Explained by Top 50 PC",
     xlab = "Number of Components", ylab = "Variance Explained")
#dev.off()

###Replace train
train.score <- as.matrix(trainnzv) %*% train.pc$rotation[,1:50]
train.pca <- cbind(label, as.data.frame(train.score))

#Explore Train after PCA by creating PC1 and PC2 plot by groups
Groups = label.factor # define grouping variable
ggplot(data=train.pca, aes(x=train.pca$PC1, y=train.pca$PC2, colour=Groups, shape=Groups)) + 
  geom_point(size=2) + 
  theme(aspect.ratio=1) + 
  scale_shape_manual(values=seq(48,57)) + 
  labs(title="Biplot of first two dimensions of the PCA solution", x="Principal Component 1 (65%)", 
       y="Principal Component 2 (21%)") + stat_ellipse(type="norm", level=0.95) + 
  geom_vline(xintercept=c(-0,0), linetype="dashed", size=0.3) + 
  geom_hline(yintercept=c(-0,0), linetype="dashed", size=0.3)

#The biplot reveals specific or groups of high or low pixel 
#values that are characteristic for distinct product groups. 
#For instance, a trouser is more likely to have dark (high) pixel-values 
#in the top part of the image whereas a 
#sandal is more likely to have low pixel-values.


###Replace validation
label <- valid[,1]
validwolabels<- valid[,-1]
keep <- setdiff(names(validwolabels), discard)
validnzv <- validwolabels[,keep]
validnzv <- validnzv / 255
valid.score <- as.matrix(validnzv) %*% train.pc$rotation[,1:50]
valid.pca <- cbind(label, as.data.frame(valid.score))
valid.pca$label <- factor(valid.pca$label)

###Create dataframe to store model results
model.accuracytest<- setNames(data.frame(matrix(ncol = 4, nrow = 0)), 
                              c("model", "accuracy_test","Multiclass AUC","SecsTaken_test"))

parallel::stopCluster(cl)


###################################################
#########Start running Models###########
cl <- parallel::makeCluster(detectCores(), type='PSOCK')
doParallel::registerDoParallel(cl)

library(pROC)
###Model with Random forest on PCA Data
library(randomForest)
set.seed(1)
start.time <- Sys.time()
fmnist_rf=randomForest(label~. ,data = train.pca, ntree=56)
t <- Sys.time() - start.time


predrf <- predict(fmnist_rf,valid.pca)
predict(fmnist_rf,valid.pca, type = "prob")

rf_cm <- confusionMatrix(predrf, 
                         valid.pca$label,
                         dnn = c("RF-Predicted", "Actual"))


#Compute Multi-class area under the curve
roc_df <- multiclass.roc(predrf,as.numeric(valid.pca$label))
roc_df$auc[1]

model.accuracytest['rf',] <- c('rf', rf_cm$overall[1],roc_df$auc[1],as.numeric(t, units = "secs"))


###Model with SVM on PCA Data

library(e1071)

set.seed(1)
start.time <- Sys.time()
fmnist_svm <- svm(label ~ ., data=train.pca)
t<- Sys.time() - start.time

pred_svm <- predict(fmnist_svm, valid.pca)

svm_cm <- confusionMatrix(pred_svm, 
                          valid.pca$label,
                          dnn = c("SVM-Predicted", "Actual"))


#Compute Multi-class area under the curve
roc_svm <- multiclass.roc(pred_svm,as.numeric(valid.pca$label))
roc_svm$auc[1]

model.accuracytest['svm',] <- c('svm', svm_cm$overall[1],roc_svm$auc[1],as.numeric(t, units = "secs"))


###############
#Model with LDA on PCA Data
#Remarks : Very quick
library(MASS)

set.seed(1)
start.time <- Sys.time()
fmnist_lda <- lda(label~.,data = train.pca)
t<- Sys.time() - start.time

pred_lda <- predict(fmnist_lda, valid.pca)

lda_cm <- confusionMatrix(pred_lda$class, 
                          valid.pca$label,
                          dnn = c("LDA-Predicted", "Actual"))

#Compute Multi-class area under the curve
roc_lda <- multiclass.roc(pred_lda$class,as.numeric(valid.pca$label))
roc_lda$auc[1]

model.accuracytest['lda',] <- c('lda', lda_cm$overall[1],roc_lda$auc[1],as.numeric(t, units = "secs"))



###Model with Neural Network--nnet--on PCA Data
#We choose no_of_nodes=120 and maxiter=130
#"softmax" should be set to TRUE when performing classification
library(nnet)

set.seed(1)
n <- names(train.pca[,-1])
f <- as.formula(paste("label ~", paste(n[!n %in% "medv"], collapse = " + ")))
start.time <- Sys.time()
fmnist_nnet <- nnet(f,data = train.pca,
                    size=120,maxit=130,MaxNWts = 80000)
t<- Sys.time() - start.time

library(NeuralNetTools)

plotnet(fmnist_nnet,skip = TRUE)

pred_nnet <- predict(fmnist_nnet,valid.pca,type="class")
nnet_cm <- confusionMatrix(factor(pred_nnet), 
                           valid.pca$label,
                           dnn = c("nnet-Predicted", "Actual"))

#Compute Multi-class area under the curve
roc_nnet <- multiclass.roc(pred_nnet,as.numeric(valid.pca$label))
roc_nnet$auc[1]

model.accuracytest['nnet',] <- c('nnet', nnet_cm$overall[1],roc_nnet$auc[1],as.numeric(t, units = "secs"))


###Model with k-nearest neighbours on PCA Data
##Remarks : Takes time to run
library(caret)
library('e1071')

trctrl <- trainControl(method = "repeatedcv", number = 4, repeats = 1, allowParallel = T)
set.seed(1)
start.time <- Sys.time()
fmnist_knn <- train(label ~., data = train.pca, method = "knn",
                    trControl=trctrl,preProcess = c("center", "scale"),
                    tuneLength = 3)
t<- Sys.time() - start.time

fmnist_knn
plot(fmnist_knn)

pred_knn <- predict(fmnist_knn, newdata = valid.pca)

knn_cm <- confusionMatrix(pred_knn, 
                          valid.pca$label,
                          dnn = c("knn-Predicted", "Actual"))

#Compute Multi-class area under the curve
roc_knn <- multiclass.roc(pred_knn,as.numeric(valid.pca$label))
roc_knn$auc[1]

model.accuracytest['knn',] <- c('knn', knn_cm$overall[1],roc_knn$auc[1],as.numeric(t, units = "secs"))

parallel::stopCluster(cl)


dev.off()
#Lift-chart and Decile chart Random Forest
actual = as.numeric(valid.pca$label)
predicted = as.numeric(predrf)
gain <- gains(actual, predicted, groups=10)
##Lift-Chart
plot(c(0, gain$cume.pct.of.total*sum(actual)) ~ c(0, gain$cume.obs),
     xlab = "# cases", ylab = "Cumulative", type="l",col = "blue",
     main = "Random Forest - Lift Chart")
lines(c(0,sum(actual))~c(0,dim(valid.pca)[1]), col="red", lty=2)

##Gain-chart
heights <- gain$mean.resp/mean(actual)
midpoints <- barplot(heights, names.arg = gain$depth, ylim = c(0 ,2.3),
                     xlab = "Percentile", ylab = "Mean Response", 
                     main = "Random Forest - Decile-wise lift chart",
                     col = "blue")
# add labels to bars on decile chart
text(midpoints, heights+0.2, labels=round(heights, 1), cex = 1)

#Lift-chart and Decile chart LDA
actual = as.numeric(valid.pca$label)
predicted = as.numeric(pred_lda$class)
gain <- gains(actual, predicted, groups=10)
##Lift-Chart
plot(c(0, gain$cume.pct.of.total*sum(actual)) ~ c(0, gain$cume.obs),
     xlab = "# cases", ylab = "Cumulative", type="l",col = "blue",
     main = "LDA - Lift Chart")
lines(c(0,sum(actual))~c(0,dim(valid.pca)[1]), col="red", lty=2)

##Gain-chart
heights <- gain$mean.resp/mean(actual)
midpoints <- barplot(heights, names.arg = gain$depth, ylim = c(0 ,2.3),
                     xlab = "Percentile", ylab = "Mean Response", 
                     main = "LDA - Decile-wise lift chart",
                     col = "blue")
# add labels to bars on decile chart
text(midpoints, heights+0.2, labels=round(heights, 1), cex = 1)



#Lift-chart and Decile chart Neural Network
actual = as.numeric(valid.pca$label)
predicted = as.numeric(pred_nnet)
gain <- gains(actual, predicted, groups=10)
##Lift-Chart
plot(c(0, gain$cume.pct.of.total*sum(actual)) ~ c(0, gain$cume.obs),
     xlab = "# cases", ylab = "Cumulative", type="l",col = "blue",
     main = "Neural Network - Lift Chart")
lines(c(0,sum(actual))~c(0,dim(valid.pca)[1]), col="red", lty=2)

##Gain-chart
heights <- gain$mean.resp/mean(actual)
midpoints <- barplot(heights, names.arg = gain$depth, ylim = c(0 ,2.3),
                     xlab = "Percentile", ylab = "Mean Response", 
                     main = "Neural Network - Decile-wise lift chart",
                     col = "blue")
# add labels to bars on decile chart
text(midpoints, heights+0.2, labels=round(heights, 1), cex = 1)


#Lift-chart and Decile chart k-Nearest Neighbours
actual = as.numeric(valid.pca$label)
predicted = as.numeric(pred_knn)
gain <- gains(actual, predicted, groups=10)
##Lift-Chart
plot(c(0, gain$cume.pct.of.total*sum(actual)) ~ c(0, gain$cume.obs),
     xlab = "# cases", ylab = "Cumulative", type="l",col = "blue",
     main = "knn - Lift Chart")
lines(c(0,sum(actual))~c(0,dim(valid.pca)[1]), col="red", lty=2)

##Gain-chart
heights <- gain$mean.resp/mean(actual)
midpoints <- barplot(heights, names.arg = gain$depth, ylim = c(0 ,2.3),
                     xlab = "Percentile", ylab = "Mean Response", 
                     main = "knn - Decile-wise lift chart",
                     col = "blue")
# add labels to bars on decile chart
text(midpoints, heights+0.2, labels=round(heights, 1), cex = 1)


#Lift-chart and Decile chart SVM
actual = as.numeric(valid.pca$label)
predicted = as.numeric(pred_svm)
gain <- gains(actual, predicted, groups=10)
##Lift-Chart
plot(c(0, gain$cume.pct.of.total*sum(actual)) ~ c(0, gain$cume.obs),
     xlab = "# cases", ylab = "Cumulative", type="l",col = "blue",
     main = "SVM - Lift Chart")
lines(c(0,sum(actual))~c(0,dim(valid.pca)[1]), col="red", lty=2)

##Gain-chart
heights <- gain$mean.resp/mean(actual)
midpoints <- barplot(heights, names.arg = gain$depth, ylim = c(0 ,2.3),
                     xlab = "Percentile", ylab = "Mean Response", 
                     main = "SVM - Decile-wise lift chart",
                     col = "blue")
# add labels to bars on decile chart
text(midpoints, heights+0.2, labels=round(heights, 1), cex = 1)



#########
####################VISUALIZE PREDICTIONS#####################3
#Visualize predictions with actual digits using SVM model
dev.off()
svm.preds <- predict(fmnist_svm, valid.pca, array.layout = "colmajor")
#svm.preds[1:5]
label.name = svm.preds
categories = c("T-shirt", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Boot")
levels(label.name) = categories # add category column (character)



##Plot images with green as correctly and red as wrong predicted class
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
    text(0.09,0.95,
         col=ifelse(preds[i]==validrep[i,1],"green","red"), 
         cex=1.2, name[i])
  }
  par(op)
}

plotResults(sample(1:length(svm.preds), 9, replace=F),
            svm.preds, label.name)

#Create a loop function for random images every sec seconds
loop <- function(sec){
  i = 1
  while(TRUE){
    if (i %% 8 == 0){
      break           #A condition to break out of the loop
    }
    plotResults(sample(1:length(svm.preds), 9, replace=F),
                svm.preds, label.name)           #Run your code
    Sys.sleep(time = sec) #Time in seconds
    
    i = i + 1
  }
}

#run loop function

loop(sec=2)

round(as.numeric(model.accuracytest$accuracy_test), digits = 2)

###Create dataframe to store model results
model_accuracyTest<- setNames(data.frame(matrix(ncol = 4, nrow = 0)), 
                              c("Model", "Accuracyt","Multiclass AUC","SecsTaken"))
model_accuracyTest$accuracy_test <- round(as.numeric(model.accuracytest$accuracy_test), digits = 2)
model_accuracyTest$`Multiclass AUC` <- round(as.numeric(model.accuracytest$`Multiclass AUC`), digits = 2)
model_accuracyTest$SecsTaken_test <- round(as.numeric(model.accuracytest$SecsTaken_test), digits = 0)
