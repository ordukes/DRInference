library(lmtest)
library(mvtnorm)
library(randomForest)
library(SuperLearner)
library(sandwich)
library(clusterGeneration)
library(np)
library(xtable)
library(glmnet)
library(ranger)
library(gam)
library(polspline)
library(KernelKnn)

source("dri_functions.R")

truth=0

#Super learner library
SL.gam.3 <- function(..., deg.gam = 3) SL.gam(..., deg.gam = deg.gam)
SL.gam.4 <- function(..., deg.gam = 4) SL.gam(..., deg.gam = deg.gam)
SL.gam.5 <- function(..., deg.gam = 5) SL.gam(..., deg.gam = deg.gam)
SL.gam.6 <- function(..., deg.gam = 6) SL.gam(..., deg.gam = deg.gam)
SL.gam.7 <- function(..., deg.gam = 7) SL.gam(..., deg.gam = deg.gam)
SL.gam.8 <- function(..., deg.gam = 8) SL.gam(..., deg.gam = deg.gam)
SL.gam.9 <- function(..., deg.gam = 9) SL.gam(..., deg.gam = deg.gam)
SL.gam.10 <- function(..., deg.gam = 10) SL.gam(..., deg.gam = deg.gam)

sl.lib=c("SL.mean","SL.glm.interaction","SL.kernelKnn",paste("SL.gam.", 3:10, sep = ""),"SL.ranger")

run.sim <- function(N,nsim,seed,sigma){
  
  thetahat=matrix(NA,nrow=nsim,45)
  
  for(i in 1:nsim){#i=1
    
    #Generate data
    set.seed(i*seed)
    l1<-runif(N,-2,2)
    l2<-rbinom(N,1,0.5)
    l<-cbind(l1,l2)
    gamma=c(-1,2) 
    beta=c(-1,2)
    
    l_int<-l1*l2
    a=rbinom(N,1,plogis(cbind(l1,l_int)%*%gamma))
    y=rnorm(N,cbind(l1,l_int)%*%beta,1)
    
    #Super-learner
    nsplits <- 5
    s <- sample(rep(1:nsplits,ceiling(N/nsplits))[1:N])
    G_sl=rep(NA,N)
    for (vfold in 1:nsplits){
      train <- s!=vfold; test <- s==vfold
      if (nsplits==1){ train <- test }
      modela_sl=SuperLearner(Y=a[train],X=data.frame(l[train,]),newX=data.frame(l[test,]),SL.library = sl.lib,family="binomial")
      G_sl[test]=modela_sl$SL.predict
    } 
    M_sl=rep(NA,N)
    for (vfold in 1:nsplits){
      train <- s!=vfold; test <- s==vfold
      if (nsplits==1){ train <- test }
      modely_sl=SuperLearner(Y=y[train],X=data.frame(l[train,]),newX=data.frame(l[test,]),SL.library = sl.lib)
      M_sl[test]=modely_sl$SL.predict
    } 
    
    #Lasso - misspecified
    G_lm<-rep(NA,N)
    lambda_init_G=lambda_select_binom(y=a,cov=l)
    for (vfold in 1:nsplits){
      train <- s!=vfold; test <- s==vfold
      if (nsplits==1){ train <- test }
      modela_lm=glmnet(l[train,],a[train],alpha=1,lambda=lambda_init_G,family="binomial")
      G_lm[test]=predict(modela_lm,l[test,],type="response")
    } 
    M_lm<-rep(NA,N)
    lambda_init_M=lambda_select(y=y,cov=cbind(a,l))
    for (vfold in 1:nsplits){
      train <- s!=vfold; test <- s==vfold
      if (nsplits==1){ train <- test }
      modely_lm=glmnet(cbind(a[train],l[train,]),y[train],alpha=1,lambda=lambda_init_M)
      M_lm[test]=predict(modely_lm,cbind(rep(0,table(test)[2]),l[test,]),type="response")
    } 
    
    # #Both nuisances consistent - super learner
    g_est<-g_estimation.test(N=N,y=y,a=a,G=G_sl)
    thetahat[i,1:3]<-g_est
    plug<-plug_in.test(N=N,y=y,a=a,M=M_sl)
    thetahat[i,4:6]<-plug
    dml_est=dml_cross.test(N=N,y=y,a=a,G=G_sl,M=M_sl)
    thetahat[i,7:9]<-dml_est
    rdml_est=rdml_cross.test(N=N,y=y,a=a,nsplits=nsplits,s=s,G=G_sl,M=M_sl)
    thetahat[i,10:12]<-rdml_est
    # 
    # #Only exposure model correct
    g_est<-g_estimation.test(N=N,y=y,a=a,G=G_sl)
    thetahat[i,13:15]<-g_est
    plug<-plug_in.test(N=N,y=y,a=a,M=M_lm)
    thetahat[i,16:18]<-plug
    dml_est=dml_cross.test(N=N,y=y,a=a,G=G_sl,M=M_lm)
    thetahat[i,19:21]<-dml_est
    rdml_est=rdml_cross.test(N=N,y=y,a=a,nsplits=nsplits,s=s,G=G_sl,M=M_lm)
    thetahat[i,22:24]<-rdml_est
    # 
    # #Only outcome model correct
    g_est<-g_estimation.test(N=N,y=y,a=a,G=G_lm)
    thetahat[i,25:27]<-g_est
    plug<-plug_in.test(N=N,y=y,a=a,M=M_sl)
    thetahat[i,28:30]<-plug
    dml_est=dml_cross.test(N=N,y=y,a=a,G=G_lm,M=M_sl)
    thetahat[i,31:33]<-dml_est
    rdml_est=rdml_cross.test(N=N,y=y,a=a,nsplits=nsplits,s=s,G=G_lm,M=M_sl)
    thetahat[i,34:36]<-rdml_est
    # 
    # #Super efficient estimators
    dml_est=dml_cross.test(N=N,y=y,a=a,G=rep(mean(a),N),M=M_sl)
    thetahat[i,37:39]<-dml_est
    rdml_est=rdml_cross.test(N=N,y=y,a=a,nsplits=nsplits,s=s,G=rep(mean(a),N),M=M_sl)
    thetahat[i,40:42]<-rdml_est
    
    #Oracle estimator
    G_or<-plogis(cbind(l1,l_int)%*%gamma)
    M_or<-cbind(l1,l_int)%*%beta
    dml_est=dml_cross.test(N=N,y=y,a=a,G=G_or,M=G_or)
    thetahat[i,43:45]<-dml_est
    
    alpha=0.05
    cat(i,round(c(mean(thetahat[1:i,1]),mean(thetahat[1:i,4]),
                  mean(thetahat[1:i,7]),mean(thetahat[1:i,10]),
                  mean(thetahat[1:i,13]),mean(thetahat[1:i,16]),
                  mean(thetahat[1:i,19]),mean(thetahat[1:i,22]),
                  mean(thetahat[1:i,25]),mean(thetahat[1:i,28]),
                  mean(thetahat[1:i,31]),mean(thetahat[1:i,34]),
                  mean(thetahat[1:i,37]),mean(thetahat[1:i,40]),
                  mean(thetahat[1:i,43]),
                  mean(thetahat[1:i,3]<alpha),mean(thetahat[1:i,6]<alpha),mean(thetahat[1:i,9]<alpha),
                  mean(thetahat[1:i,12]<alpha),mean(thetahat[1:i,15]<alpha),
                  mean(thetahat[1:i,18]<alpha),mean(thetahat[1:i,21]<alpha),
                  mean(thetahat[1:i,24]<alpha),mean(thetahat[1:i,27]<alpha),
                  mean(thetahat[1:i,30]<alpha),mean(thetahat[1:i,33]<alpha),
                  mean(thetahat[1:i,36]<alpha),mean(thetahat[1:i,39]<alpha),
                  mean(thetahat[1:i,42]<alpha),mean(thetahat[1:i,45]<alpha)),digits=3),"\n")
    
    #cat(i,round(thetahat[i,],digits=2),"\n")
    
  }
  return(thetahat)
}

result250<-run.sim(N=250,nsim=1000,seed=15)
save.image("results_exp1")
result500<-run.sim(N=500,nsim=1000,seed=125)
save.image("results_exp1")
result1000<-run.sim(N=1000,nsim=1000,seed=198)
save.image("results_exp1")
result2000<-run.sim(N=2000,nsim=1000,seed=3)
save.image("results_exp1")
result3000<-run.sim(N=3000,nsim=1000,seed=54)
save.image("results_exp1")
result5000<-run.sim(N=5000,nsim=1000,seed=220)

load("results_exp1")

sequence<-c(1,4,7,10,13,16,19,22,25,28,31,34,37,40,43)
bias_mat<-matrix(NA,6,length(sequence))
for(z in sequence){
  q<-(z+2)/3
  bias_mat[1,q]=mean(result250[,z],na.rm=TRUE)/sqrt(250)
  bias_mat[2,q]=mean(result500[,z],na.rm=TRUE)/sqrt(500)
  bias_mat[3,q]=mean(result1000[,z],na.rm=TRUE)/sqrt(1000)
  bias_mat[4,q]=mean(result2000[,z],na.rm=TRUE)/sqrt(2000)
  bias_mat[5,q]=mean(result3000[,z],na.rm=TRUE)/sqrt(3000)
  bias_mat[6,q]=mean(result5000[,z],na.rm=TRUE)/sqrt(5000)
}

sqrtn_bias_mat<-matrix(NA,6,length(sequence))
for(z in sequence){
  q<-(z+2)/3
  sqrtn_bias_mat[1,q]=mean(result250[,z],na.rm=TRUE)
  sqrtn_bias_mat[2,q]=mean(result500[,z],na.rm=TRUE)
  sqrtn_bias_mat[3,q]=mean(result1000[,z],na.rm=TRUE)
  sqrtn_bias_mat[4,q]=mean(result2000[,z],na.rm=TRUE)
  sqrtn_bias_mat[5,q]=mean(result3000[,z],na.rm=TRUE)
  sqrtn_bias_mat[6,q]=mean(result5000[,z],na.rm=TRUE)
}

var_mat<-matrix(NA,6,length(sequence))
for(z in sequence){
  q<-(z+2)/3
  var_mat[1,q]=var(result250[,z]/sqrt(250),na.rm=TRUE)/mean(result250[,z+1]^2,na.rm=TRUE)
  var_mat[2,q]=var(result500[,z]/sqrt(500),na.rm=TRUE)/mean(result500[,z+1]^2,na.rm=TRUE)
  var_mat[3,q]=var(result1000[,z]/sqrt(1000),na.rm=TRUE)/mean(result1000[,z+1]^2,na.rm=TRUE)
  var_mat[4,q]=var(result2000[,z]/sqrt(2000),na.rm=TRUE)/mean(result2000[,z+1]^2,na.rm=TRUE)
  var_mat[5,q]=var(result3000[,z]/sqrt(3000),na.rm=TRUE)/mean(result3000[,z+1]^2,na.rm=TRUE)
  var_mat[6,q]=var(result5000[,z]/sqrt(5000),na.rm=TRUE)/mean(result5000[,z+1]^2,na.rm=TRUE)
}

sequence<-c(3,6,9,12,15,18,21,24,27,30,33,36,39,42,45)
size_mat<-matrix(NA,6,length(sequence))
alpha=0.05
for(z in sequence){
  q<-(z)/3
  size_mat[1,q]=mean(result250[,z]<alpha,na.rm=TRUE)
  size_mat[2,q]=mean(result500[,z]<alpha,na.rm=TRUE)
  size_mat[3,q]=mean(result1000[,z]<alpha,na.rm=TRUE)
  size_mat[4,q]=mean(result2000[,z]<alpha,na.rm=TRUE)
  size_mat[5,q]=mean(result3000[,z]<alpha,na.rm=TRUE)
  size_mat[6,q]=mean(result5000[,z]<alpha,na.rm=TRUE)
}

#Tables
i=1
ssize<-c(250,500,1000,2000,3000,5000)
pdf(paste("exp1_1.pdf",sep=""))
par(mfrow=c(2,2))
plot(ssize,bias_mat[1:6,i],ylim=c(min(bias_mat[1:6,(i):(i+3)])-0.01,max(bias_mat[1:6,(i):(i+3)])+0.01),
     pch=16,ylab="Bias",xlab="n",col=1)
points(ssize,bias_mat[1:6,i+1],pch=15,col=2)
points(ssize,bias_mat[1:6,i+2],pch=17,col=3)
points(ssize,bias_mat[1:6,i+3],pch=18,col=4)
lines(ssize,bias_mat[1:6,i],lty=3,col=1)
lines(ssize,bias_mat[1:6,i+1],lty=3,col=2)
lines(ssize,bias_mat[1:6,i+2],lty=3,col=3)
lines(ssize,bias_mat[1:6,i+3],lty=3,col=4)
lines(ssize,rep(0,6),lty=2,lwd=0.7,col="darkgray")
axis(side=1,at=ssize,labels=T)
ey<-expression(n^{1/2}%*%Bias)
plot(ssize,sqrtn_bias_mat[1:6,i],ylim=c(min(sqrtn_bias_mat[1:6,(i):(i+3)])-0.2,max(sqrtn_bias_mat[1:6,(i):(i+3)])+0.2),
     pch=16,col=1,ylab=ey,xlab="n")
points(ssize,sqrtn_bias_mat[1:6,i+1],pch=15,col=2)
points(ssize,sqrtn_bias_mat[1:6,i+2],pch=17,col=3)
points(ssize,sqrtn_bias_mat[1:6,i+3],pch=18,col=4)
lines(ssize,sqrtn_bias_mat[1:6,i],lty=3,col=1)
lines(ssize,sqrtn_bias_mat[1:6,i+1],lty=3,col=2)
lines(ssize,sqrtn_bias_mat[1:6,i+2],lty=3,col=3)
lines(ssize,sqrtn_bias_mat[1:6,i+3],lty=3,col=4)
lines(ssize,rep(0,6),lty=2,lwd=0.7,col="darkgray")
axis(side=1,at=ssize,labels=T)
plot(ssize,size_mat[1:6,i],ylim=c(0,0.4),
     pch=16,col=1,ylab="Size",xlab="n")
points(ssize,size_mat[1:6,i+1],pch=15,col=2)
points(ssize,size_mat[1:6,i+2],pch=17,col=3)
points(ssize,size_mat[1:6,i+3],pch=17,col=4)
lines(ssize,size_mat[1:6,i],lty=3,col=1)
lines(ssize,size_mat[1:6,i+1],lty=3,col=2)
lines(ssize,size_mat[1:6,i+2],lty=3,col=3)
lines(ssize,size_mat[1:6,i+3],lty=3,col=4)
lines(ssize,rep(0.05,6),lty=2,lwd=0.7,col="darkgray")
axis(side=1,at=ssize,labels=T)
ey2<-"MC var/est var"
plot(ssize,var_mat[1:6,i],ylim=c(min(var_mat[1:6,(i):(i+3)])-0.2,max(var_mat[1:6,(i):(i+3)])+0.2),
     pch=16,col=1,ylab=ey2,xlab="n")
points(ssize,var_mat[1:6,i+1],pch=15,col=2)
points(ssize,var_mat[1:6,i+2],pch=17,col=3)
points(ssize,var_mat[1:6,i+3],pch=18,col=4)
lines(ssize,var_mat[1:6,i],lty=3,col=1)
lines(ssize,var_mat[1:6,i+1],lty=3,col=2)
lines(ssize,var_mat[1:6,i+2],lty=3,col=3)
lines(ssize,var_mat[1:6,i+3],lty=3,col=4)
lines(ssize,rep(1,6),lty=2,lwd=0.7,col="darkgray")
axis(side=1,at=ssize,labels=T)
dev.off()

i=5
pdf(paste("exp1_2.pdf",sep=""))
par(mfrow=c(2,2))
plot(ssize,bias_mat[1:6,i],ylim=c(min(bias_mat[1:6,(i):(i+3)])-0.01,max(bias_mat[1:6,i])+0.01),
     pch=16,ylab="Bias",xlab="n",col=1)
points(ssize,bias_mat[1:6,i+2],pch=17,col=3)
points(ssize,bias_mat[1:6,i+3],pch=18,col=4)
lines(ssize,bias_mat[1:6,i],lty=3,col=1)
lines(ssize,bias_mat[1:6,i+2],lty=3,col=3)
lines(ssize,bias_mat[1:6,i+3],lty=3,col=4)
lines(ssize,rep(0,6),lty=2,lwd=0.7,col="darkgray")
axis(side=1,at=ssize,labels=T)
ey<-expression(n^{1/2}%*%Bias)
plot(ssize,sqrtn_bias_mat[1:6,i],ylim=c(min(sqrtn_bias_mat[1:6,(i):(i+3)])-0.2,max(sqrtn_bias_mat[1:6,i])+0.2),
     pch=16,col=1,ylab=ey,xlab="n")
points(ssize,sqrtn_bias_mat[1:6,i+2],pch=17,col=3)
points(ssize,sqrtn_bias_mat[1:6,i+3],pch=18,col=4)
lines(ssize,sqrtn_bias_mat[1:6,i],lty=3,col=1)
lines(ssize,sqrtn_bias_mat[1:6,i+2],lty=3,col=3)
lines(ssize,sqrtn_bias_mat[1:6,i+3],lty=3,col=4)
lines(ssize,rep(0,6),lty=2,lwd=0.7,col="darkgray")
axis(side=1,at=ssize,labels=T)
plot(ssize,size_mat[1:6,i],ylim=c(0,0.15),
     pch=16,col=1,ylab="Size",xlab="n")
points(ssize,size_mat[1:6,i+2],pch=17,col=3)
points(ssize,size_mat[1:6,i+3],pch=17,col=4)
lines(ssize,size_mat[1:6,i],lty=3,col=1)
lines(ssize,size_mat[1:6,i+2],lty=3,col=3)
lines(ssize,size_mat[1:6,i+3],lty=3,col=4)
lines(ssize,rep(0.05,6),lty=2,lwd=0.7,col="darkgray")
axis(side=1,at=ssize,labels=T)
ey2<-"MC var/est var"
plot(ssize,var_mat[1:6,i],ylim=c(min(var_mat[1:6,(i):(i+3)])-0.2,max(var_mat[1:6,(i):(i+3)])+0.2),
     pch=16,col=1,ylab=ey2,xlab="n")
points(ssize,var_mat[1:6,i+2],pch=17,col=3)
points(ssize,var_mat[1:6,i+3],pch=18,col=4)
lines(ssize,var_mat[1:6,i],lty=3,col=1)
lines(ssize,var_mat[1:6,i+2],lty=3,col=3)
lines(ssize,var_mat[1:6,i+3],lty=3,col=4)
lines(ssize,rep(1,6),lty=2,lwd=0.7,col="darkgray")
axis(side=1,at=ssize,labels=T)
dev.off()

i=9
pdf(paste("exp1_3.pdf",sep=""))
par(mfrow=c(2,2))
plot(ssize,bias_mat[1:6,i+1],ylim=c(min(bias_mat[1:6,(i):(i+3)])-0.01,max(bias_mat[1:6,i+1])+0.01),
     pch=15,ylab="Bias",xlab="n",col=2)
points(ssize,bias_mat[1:6,i+2],pch=17,col=3)
points(ssize,bias_mat[1:6,i+3],pch=18,col=4)
lines(ssize,bias_mat[1:6,i+1],lty=3,col=2)
lines(ssize,bias_mat[1:6,i+2],lty=3,col=3)
lines(ssize,bias_mat[1:6,i+3],lty=3,col=4)
lines(ssize,rep(0,6),lty=2,lwd=0.7,col="darkgray")
axis(side=1,at=ssize,labels=T)
ey<-expression(n^{1/2}%*%Bias)
plot(ssize,sqrtn_bias_mat[1:6,i+1],ylim=c(min(sqrtn_bias_mat[1:6,(i):(i+3)])-0.2,max(sqrtn_bias_mat[1:6,i+1])+0.2),
     pch=15,col=2,ylab=ey,xlab="n")
points(ssize,sqrtn_bias_mat[1:6,i+2],pch=17,col=3)
points(ssize,sqrtn_bias_mat[1:6,i+3],pch=18,col=4)
lines(ssize,sqrtn_bias_mat[1:6,i+1],lty=3,col=2)
lines(ssize,sqrtn_bias_mat[1:6,i+2],lty=3,col=3)
lines(ssize,sqrtn_bias_mat[1:6,i+3],lty=3,col=4)
lines(ssize,rep(0,6),lty=2,lwd=0.7,col="darkgray")
axis(side=1,at=ssize,labels=T)
plot(ssize,size_mat[1:6,i+1],ylim=c(0,0.4),
     pch=15,col=2,ylab="Size",xlab="n")
points(ssize,size_mat[1:6,i+2],pch=17,col=3)
points(ssize,size_mat[1:6,i+3],pch=17,col=4)
#lines(ssize,size_mat[1:6,i],lty=3,col=1)
lines(ssize,size_mat[1:6,i+1],lty=3,col=2)
lines(ssize,size_mat[1:6,i+2],lty=3,col=3)
lines(ssize,size_mat[1:6,i+3],lty=3,col=4)
lines(ssize,rep(0.05,6),lty=2,lwd=0.7,col="darkgray")
axis(side=1,at=ssize,labels=T)
ey2<-"MC var/est var"
plot(ssize,var_mat[1:6,i+1],ylim=c(min(var_mat[1:6,(i):(i+3)])-0.2,max(var_mat[1:6,(i):(i+3)])+0.2),
     pch=15,col=2,ylab=ey2,xlab="n")
points(ssize,var_mat[1:6,i+2],pch=17,col=3)
points(ssize,var_mat[1:6,i+3],pch=18,col=4)
lines(ssize,var_mat[1:6,i+1],lty=3,col=2)
lines(ssize,var_mat[1:6,i+2],lty=3,col=3)
lines(ssize,var_mat[1:6,i+3],lty=3,col=4)
lines(ssize,rep(1,6),lty=2,lwd=0.7,col="darkgray")
axis(side=1,at=ssize,labels=T)
dev.off()
