# install.packages("tree")     # install package

library(tree)



### Read Dataset

dat <- read.table("d:/letter-recognition.data.txt",sep = ",")

head(dat)  ## See dataset

dat <- dat[1:2000,] ## Use 2000 data

train = 1:1500      ## Use 1500 data for training model

mis = list()        ## List of misclassification rate for each model



### Tree Model

tree =  tree(V1~., data= dat[train,])

summary(tree)



plot(tree)

text(tree)



pred.tree = predict(tree,newdata = dat[-train,],type="class")

summary(pred.tree)



## Confusion Matrix

tree.table = table(dat[-train,]$V1, pred.tree)



mis$tree = 1 - sum(diag(tree.table))/sum(tree.table)



## Prune Tree

treecv = cv.tree(tree,FUN =prune.misclass,K=5)

treecv

prunetree = prune.misclass(tree,best=11)



pred.cvtree = predict(prunetree,newdata = dat[-train,],type="class")

prunetree.table = table(dat[-train,]$V1, pred.cvtree)

mis$cvtree = 1 - sum(diag(prunetree.table))/sum(prunetree.table)



#### Random Forest



# install.packages("randomForest")

library("randomForest")



rantree26 = randomForest(V1~., data= dat[train,],mtry=26,importance=T)

pred.rantree26 = predict(rantree26,newdata=dat[-train,],type="class")

rantree26.table = table(dat[-train,]$V1,pred.rantree26)

mis$rantree26 = 1 - sum(diag(rantree26.table))/sum(rantree26.table)





rantree5 = randomForest(V1~., data= dat[train,],mtry=5,importance=T)

pred.rantree5 = predict(rantree5,newdata=dat[-train,])

rantree5.table = table(dat[-train,]$V1,pred.rantree5)

mis$rantree5 = 1 - sum(diag(rantree5.table))/sum(rantree5.table)







#### Boosting



# install.packages("gbm")

library(gbm)



boost = gbm(V1~., data=dat[train,],n.trees = 2000,interaction.depth = 5)

summary(boost)



pred.boost = predict(boost,newdata=dat[-train,],n.trees = 2000,type="link")

boostclass = apply(pred.boost,1,which.max)

boost.table = table(dat[-train,]$V1,boostclass)

mis$boost = 1 - sum(diag(boost.table))/sum(boost.table)

