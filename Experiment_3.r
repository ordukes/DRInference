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
library(grf)
library(KernelKnn)
library(hdm)

rm(list=ls())

source("dri_functions.R")

run.sim <- function(N,nsim,seed,sigma,c_gamma,c_beta){
  
  thetahat=matrix(NA,nrow=nsim,27)
  
  for(i in 1:nsim){#i=1
    set.seed(i*seed)
    p<-200  
    seq_num<-0.5^c(0:(p-1))
    tp<-toeplitz(seq_num)
    l <- rmvnorm(n=N, mean=rep(0,p),sigma=tp)
    gamma<-c_gamma*(1/c(1:p))^2
    beta<-c_beta*(1/c(1:p))^2
    a<-rnorm(N,l%*%gamma,1)
    y<-rnorm(N,l%*%beta,1)
    
    nsplits <- 5
    s <- sample(rep(1:nsplits,ceiling(N/nsplits))[1:N])
    G_lm<-rep(NA,N)
    for (vfold in 1:nsplits){
      train <- s!=vfold; test <- s==vfold
      if (nsplits==1){ train <- test }
      modela_lm<-rlasso(a[train]~l[train,])
      G_lm[test]=predict(modela_lm,l[test,])
    }
    M_lm<-rep(NA,N)
    for (vfold in 1:nsplits){
      train <- s!=vfold; test <- s==vfold
      if (nsplits==1){ train <- test }
      modely_lm<-rlasso(y[train]~l[train,])
      M_lm[test]=predict(modely_lm,l[test,])
    }
    
    tau_G<-0.1
    G_slow<-G_lm-rnorm(N,mean=3*(1/((N)^tau_G)),sd=1/((N)^tau_G))
    G_true<-l%*%gamma;M_true<-l%*%beta #UPDATE - may explain change in results
    
    dml_est=dml_cross.test(N=N,y=y,a=a,G=G_lm,M=M_lm)
    thetahat[i,1:3]<-dml_est
    rdml_est=rdml_cross.test(N=N,y=y,a=a,nsplits=nsplits,s=s,G=G_lm,M=M_lm)
    thetahat[i,4:6]<-rdml_est

    #Slow PS
    dml_est=dml_cross.test(N=N,y=y,a=a,G=G_slow,M=M_lm)
    thetahat[i,7:9]<-dml_est
    rdml_est=rdml_cross.test(N=N,y=y,a=a,nsplits=nsplits,s=s,G=G_slow,M_lm)
    thetahat[i,10:12]<-rdml_est
    
    # #Misspecification
    G_m1<-rep(NA,N)
    l_red<-l[,10:p]
    for (vfold in 1:nsplits){
      train <- s!=vfold; test <- s==vfold
      if (nsplits==1){ train <- test }
      modela_lm<-rlasso(a[train]~l_red[train,])
      G_m1[test]=predict(modela_lm,l_red[test,])
    }
    dml_est=dml_cross.test(N=N,y=y,a=a,G=G_m1,M=M_lm)
    thetahat[i,13:15]<-dml_est
    rdml_est=rdml_cross.test(N=N,y=y,a=a,nsplits=nsplits,s=s,G=G_m1,M=M_lm)
    thetahat[i,16:18]<-rdml_est

    G_m2<-rep(mean(a),N)
    dml_est=dml_cross.test(N=N,y=y,a=a,G=G_m2,M=M_lm)
    thetahat[i,19:21]<-dml_est
    rdml_est=rdml_cross.test(N=N,y=y,a=a,nsplits=nsplits,s=s,G=G_m2,M=M_lm)
    thetahat[i,22:24]<-rdml_est
    
    dml_est=dml_cross.test(N=N,y=y,a=a,G=G_true,M=M_true)
    thetahat[i,25:27]<-dml_est
    
    alpha=0.05
    cat(i,round(c(mean(thetahat[1:i,1]),mean(thetahat[1:i,4]),
                  mean(thetahat[1:i,7]),mean(thetahat[1:i,10]),
                  mean(thetahat[1:i,13]),mean(thetahat[1:i,16]),
                  mean(thetahat[1:i,19]),mean(thetahat[1:i,22]),
                  mean(thetahat[1:i,3]<alpha),mean(thetahat[1:i,6]<alpha),mean(thetahat[1:i,9]<alpha),
                  mean(thetahat[1:i,12]<alpha),mean(thetahat[1:i,15]<alpha),
                  mean(thetahat[1:i,18]<alpha),mean(thetahat[1:i,21]<alpha),
                  mean(thetahat[1:i,24]<alpha)),digits=3),"\n")
    
  }
  return(thetahat)
}

sim100a<-run.sim(N=100,nsim=1000,seed=53,c_gamma=0.82,c_beta=0.82)
sim250a<-run.sim(N=250,nsim=1000,seed=339,c_gamma=0.82,c_beta=0.82)
sim500a<-run.sim(N=500,nsim=1000,seed=28,c_gamma=0.82,c_beta=0.82)
sim1000a<-run.sim(N=1000,nsim=1000,seed=8765,c_gamma=0.82,c_beta=0.82)

sim100b<-run.sim(N=100,nsim=1000,seed=382,c_gamma=0.82,c_beta=0.2)
sim250b<-run.sim(N=250,nsim=1000,seed=2,c_gamma=0.82,c_beta=0.2)
sim500b<-run.sim(N=500,nsim=1000,seed=238727,c_gamma=0.82,c_beta=0.2)
sim1000b<-run.sim(N=1000,nsim=1000,seed=98763,c_gamma=0.82,c_beta=0.2)

sim100c<-run.sim(N=100,nsim=1000,seed=9323,c_gamma=0.2,c_beta=0.82)
sim250c<-run.sim(N=250,nsim=1000,seed=38273,c_gamma=0.2,c_beta=0.82)
sim500c<-run.sim(N=500,nsim=1000,seed=9927,c_gamma=0.2,c_beta=0.82)
sim1000c<-run.sim(N=1000,nsim=1000,seed=28,c_gamma=0.2,c_beta=0.82)

save.image("results_exp3")
load("results_exp3")

#Sequence a

sequence<-c(1,4,7,10,13,16,19,22)
bias_mat<-matrix(NA,4,length(sequence))
for(z in sequence){
  q<-(z+2)/3
  bias_mat[1,q]=mean(sim100a[,z],na.rm=TRUE)/sqrt(100)
  bias_mat[2,q]=mean(sim250a[,z],na.rm=TRUE)/sqrt(250)
  bias_mat[3,q]=mean(sim500a[,z],na.rm=TRUE)/sqrt(500)
  bias_mat[4,q]=mean(sim1000a[,z],na.rm=TRUE)/sqrt(1000)
}

sqrtn_bias_mat<-matrix(NA,4,length(sequence))
for(z in sequence){
  q<-(z+2)/3
  sqrtn_bias_mat[1,q]=mean(sim100a[,z],na.rm=TRUE)
  sqrtn_bias_mat[2,q]=mean(sim250a[,z],na.rm=TRUE)
  sqrtn_bias_mat[3,q]=mean(sim500a[,z],na.rm=TRUE)
  sqrtn_bias_mat[4,q]=mean(sim1000a[,z],na.rm=TRUE)
}

var_mat<-matrix(NA,4,length(sequence))
for(z in sequence){
  q<-(z+2)/3
  var_mat[1,q]=var(sim100a[,z]/sqrt(100),na.rm=TRUE)/mean(sim100a[,z+1]^2,na.rm=TRUE)
  var_mat[2,q]=var(sim250a[,z]/sqrt(250),na.rm=TRUE)/mean(sim250a[,z+1]^2,na.rm=TRUE)
  var_mat[3,q]=var(sim500a[,z]/sqrt(500),na.rm=TRUE)/mean(sim500a[,z+1]^2,na.rm=TRUE)
  var_mat[4,q]=var(sim1000a[,z]/sqrt(1000),na.rm=TRUE)/mean(sim1000a[,z+1]^2,na.rm=TRUE)
}

sequence<-c(3,6,9,12,15,18,21,24)
size_mat<-matrix(NA,4,length(sequence))
alpha=0.05
for(z in sequence){
  q<-(z)/3
  size_mat[1,q]=mean(sim100a[,z]<alpha,na.rm=TRUE)
  size_mat[2,q]=mean(sim250a[,z]<alpha,na.rm=TRUE)
  size_mat[3,q]=mean(sim500a[,z]<alpha,na.rm=TRUE)
  size_mat[4,q]=mean(sim1000a[,z]<alpha,na.rm=TRUE)
}

#Tables
ssize<-c(100,250,500,1000)
sequence_n<-c(2,4,6,8)
pdf(paste("exp3_1.pdf",sep=""))
par(mfrow=c(2,2))
plot(ssize,bias_mat[1:4,2],ylim=c(min(bias_mat[1:4,sequence_n])-0.01,max(bias_mat[1:4,sequence_n])+0.01),
     pch=16,ylab="Bias",xlab="n",col=1)
points(ssize,bias_mat[1:4,4],pch=1,col="slategrey")
points(ssize,bias_mat[1:4,6],pch=2,col="slategrey")
points(ssize,bias_mat[1:4,8],pch=3,col=1)
lines(ssize,bias_mat[1:4,2],lty=3,col=1)
lines(ssize,bias_mat[1:4,4],lty=3,col="slategrey")
lines(ssize,bias_mat[1:4,6],lty=3,col="slategrey")
lines(ssize,bias_mat[1:4,8],lty=3,col=1)
lines(ssize,rep(0,4),lty=2,lwd=0.7,col="darkgray")
ey<-expression(n^{1/2}%*%Bias)
plot(ssize,sqrtn_bias_mat[1:4,2],ylim=c(min(sqrtn_bias_mat[1:4,sequence_n])-0.2,max(sqrtn_bias_mat[1:4,sequence_n])+0.2),
     pch=16,col=1,ylab=ey,xlab="n")
points(ssize,sqrtn_bias_mat[1:4,4],pch=1,col="slategrey")
points(ssize,sqrtn_bias_mat[1:4,6],pch=2,col="slategrey")
points(ssize,sqrtn_bias_mat[1:4,8],pch=3,col=1)
lines(ssize,sqrtn_bias_mat[1:4,2],lty=3,col=1)
lines(ssize,sqrtn_bias_mat[1:4,4],lty=3,col="slategrey")
lines(ssize,sqrtn_bias_mat[1:4,6],lty=3,col="slategrey")
lines(ssize,sqrtn_bias_mat[1:4,8],lty=3,col=1)
lines(ssize,rep(0,4),lty=2,lwd=0.7,col="darkgrey")
plot(ssize,size_mat[1:4,2],ylim=c(0,max(size_mat[1:4,sequence_n])+0.1),
     pch=16,col=1,ylab="Size",xlab="n")
points(ssize,size_mat[1:4,4],pch=1,col="slategrey")
points(ssize,size_mat[1:4,6],pch=2,col="slategrey")
points(ssize,size_mat[1:4,8],pch=3,col=1)
lines(ssize,size_mat[1:4,2],lty=3,col=1)
lines(ssize,size_mat[1:4,4],lty=3,col="slategrey")
lines(ssize,size_mat[1:4,6],lty=3,col="slategrey")
lines(ssize,size_mat[1:4,8],lty=3,col=1)
lines(ssize,rep(0.05,4),lty=2,lwd=0.7,col="darkgray")
ey2<-"MC var/est var"
plot(ssize,var_mat[1:4,2],ylim=c(min(var_mat[1:4,sequence_n])-0.2,max(var_mat[1:4,sequence_n])+0.2),
     pch=16,col=1,ylab=ey2,xlab="n")
points(ssize,var_mat[1:4,4],pch=1,col="slategrey")
points(ssize,var_mat[1:4,6],pch=2,col="slategrey")
points(ssize,var_mat[1:4,8],pch=3,col=1)
lines(ssize,var_mat[1:4,2],lty=3,col=1)
lines(ssize,var_mat[1:4,4],lty=3,col="slategrey")
lines(ssize,var_mat[1:4,6],lty=3,col="slategrey")
lines(ssize,var_mat[1:4,8],lty=3,col=1)
lines(ssize,rep(1,4),lty=2,lwd=0.7,col="darkgray")
dev.off()

#Sequence b

sequence<-c(1,4,7,10,13,16,19,22)
bias_mat<-matrix(NA,4,length(sequence))
for(z in sequence){
  q<-(z+2)/3
  bias_mat[1,q]=mean(sim100b[,z],na.rm=TRUE)/sqrt(100)
  bias_mat[2,q]=mean(sim250b[,z],na.rm=TRUE)/sqrt(250)
  bias_mat[3,q]=mean(sim500b[,z],na.rm=TRUE)/sqrt(500)
  bias_mat[4,q]=mean(sim1000b[,z],na.rm=TRUE)/sqrt(1000)
}

sqrtn_bias_mat<-matrix(NA,4,length(sequence))
for(z in sequence){
  q<-(z+2)/3
  sqrtn_bias_mat[1,q]=mean(sim100b[,z],na.rm=TRUE)
  sqrtn_bias_mat[2,q]=mean(sim250b[,z],na.rm=TRUE)
  sqrtn_bias_mat[3,q]=mean(sim500b[,z],na.rm=TRUE)
  sqrtn_bias_mat[4,q]=mean(sim1000b[,z],na.rm=TRUE)
}

var_mat<-matrix(NA,4,length(sequence))
for(z in sequence){
  q<-(z+2)/3
  var_mat[1,q]=var(sim100b[,z]/sqrt(100),na.rm=TRUE)/mean(sim100b[,z+1]^2,na.rm=TRUE)
  var_mat[2,q]=var(sim250b[,z]/sqrt(250),na.rm=TRUE)/mean(sim250b[,z+1]^2,na.rm=TRUE)
  var_mat[3,q]=var(sim500b[,z]/sqrt(500),na.rm=TRUE)/mean(sim500b[,z+1]^2,na.rm=TRUE)
  var_mat[4,q]=var(sim1000b[,z]/sqrt(1000),na.rm=TRUE)/mean(sim1000b[,z+1]^2,na.rm=TRUE)
}

sequence<-c(3,6,9,12,15,18,21,24)
size_mat<-matrix(NA,4,length(sequence))
alpha=0.05
for(z in sequence){
  q<-(z)/3
  size_mat[1,q]=mean(sim100b[,z]<alpha,na.rm=TRUE)
  size_mat[2,q]=mean(sim250b[,z]<alpha,na.rm=TRUE)
  size_mat[3,q]=mean(sim500b[,z]<alpha,na.rm=TRUE)
  size_mat[4,q]=mean(sim1000b[,z]<alpha,na.rm=TRUE)
}

pdf(paste("exp3_2.pdf",sep=""))
par(mfrow=c(2,2))
plot(ssize,bias_mat[1:4,2],ylim=c(min(bias_mat[1:4,sequence_n])-0.01,max(bias_mat[1:4,sequence_n])+0.01),
     pch=16,ylab="Bias",xlab="n",col=1)
points(ssize,bias_mat[1:4,4],pch=1,col="slategrey")
points(ssize,bias_mat[1:4,6],pch=2,col="slategrey")
points(ssize,bias_mat[1:4,8],pch=3,col=1)
lines(ssize,bias_mat[1:4,2],lty=3,col=1)
lines(ssize,bias_mat[1:4,4],lty=3,col="slategrey")
lines(ssize,bias_mat[1:4,6],lty=3,col="slategrey")
lines(ssize,bias_mat[1:4,8],lty=3,col=1)
lines(ssize,rep(0,4),lty=2,lwd=0.7,col="darkgray")
ey<-expression(n^{1/2}%*%Bias)
plot(ssize,sqrtn_bias_mat[1:4,2],ylim=c(min(sqrtn_bias_mat[1:4,sequence_n])-0.2,max(sqrtn_bias_mat[1:4,sequence_n])+0.2),
     pch=16,col=1,ylab=ey,xlab="n")
points(ssize,sqrtn_bias_mat[1:4,4],pch=1,col="slategrey")
points(ssize,sqrtn_bias_mat[1:4,6],pch=2,col="slategrey")
points(ssize,sqrtn_bias_mat[1:4,8],pch=3,col=1)
lines(ssize,sqrtn_bias_mat[1:4,2],lty=3,col=1)
lines(ssize,sqrtn_bias_mat[1:4,4],lty=3,col="slategrey")
lines(ssize,sqrtn_bias_mat[1:4,6],lty=3,col="slategrey")
lines(ssize,sqrtn_bias_mat[1:4,8],lty=3,col=1)
lines(ssize,rep(0,4),lty=2,lwd=0.7,col="darkgrey")
plot(ssize,size_mat[1:4,2],ylim=c(0,max(size_mat[1:4,sequence_n])+0.1),
     pch=16,col=1,ylab="Size",xlab="n")
points(ssize,size_mat[1:4,4],pch=1,col="slategrey")
points(ssize,size_mat[1:4,6],pch=2,col="slategrey")
points(ssize,size_mat[1:4,8],pch=3,col=1)
lines(ssize,size_mat[1:4,2],lty=3,col=1)
lines(ssize,size_mat[1:4,4],lty=3,col="slategrey")
lines(ssize,size_mat[1:4,6],lty=3,col="slategrey")
lines(ssize,size_mat[1:4,8],lty=3,col=1)
lines(ssize,rep(0.05,4),lty=2,lwd=0.7,col="darkgray")
ey2<-"MC var/est var"
plot(ssize,var_mat[1:4,2],ylim=c(min(var_mat[1:4,sequence_n])-0.2,max(var_mat[1:4,sequence_n])+0.2),
     pch=16,col=1,ylab=ey2,xlab="n")
points(ssize,var_mat[1:4,4],pch=1,col="slategrey")
points(ssize,var_mat[1:4,6],pch=2,col="slategrey")
points(ssize,var_mat[1:4,8],pch=3,col=1)
lines(ssize,var_mat[1:4,2],lty=3,col=1)
lines(ssize,var_mat[1:4,4],lty=3,col="slategrey")
lines(ssize,var_mat[1:4,6],lty=3,col="slategrey")
lines(ssize,var_mat[1:4,8],lty=3,col=1)
lines(ssize,rep(1,4),lty=2,lwd=0.7,col="darkgray")
dev.off()

#Sequence c

sequence<-c(1,4,7,10,13,16,19,22)
bias_mat<-matrix(NA,4,length(sequence))
for(z in sequence){
  q<-(z+2)/3
  bias_mat[1,q]=mean(sim100c[,z],na.rm=TRUE)/sqrt(100)
  bias_mat[2,q]=mean(sim250c[,z],na.rm=TRUE)/sqrt(250)
  bias_mat[3,q]=mean(sim500c[,z],na.rm=TRUE)/sqrt(500)
  bias_mat[4,q]=mean(sim1000c[,z],na.rm=TRUE)/sqrt(1000)
}

sqrtn_bias_mat<-matrix(NA,4,length(sequence))
for(z in sequence){
  q<-(z+2)/3
  sqrtn_bias_mat[1,q]=mean(sim100c[,z],na.rm=TRUE)
  sqrtn_bias_mat[2,q]=mean(sim250c[,z],na.rm=TRUE)
  sqrtn_bias_mat[3,q]=mean(sim500c[,z],na.rm=TRUE)
  sqrtn_bias_mat[4,q]=mean(sim1000c[,z],na.rm=TRUE)
}

var_mat<-matrix(NA,4,length(sequence))
for(z in sequence){
  q<-(z+2)/3
  var_mat[1,q]=var(sim100c[,z]/sqrt(100),na.rm=TRUE)/mean(sim100c[,z+1]^2,na.rm=TRUE)
  var_mat[2,q]=var(sim250c[,z]/sqrt(250),na.rm=TRUE)/mean(sim250c[,z+1]^2,na.rm=TRUE)
  var_mat[3,q]=var(sim500c[,z]/sqrt(500),na.rm=TRUE)/mean(sim500c[,z+1]^2,na.rm=TRUE)
  var_mat[4,q]=var(sim1000c[,z]/sqrt(1000),na.rm=TRUE)/mean(sim1000c[,z+1]^2,na.rm=TRUE)
}

sequence<-c(3,6,9,12,15,18,21,24)
size_mat<-matrix(NA,4,length(sequence))
alpha=0.05
for(z in sequence){
  q<-(z)/3
  size_mat[1,q]=mean(sim100c[,z]<alpha,na.rm=TRUE)
  size_mat[2,q]=mean(sim250c[,z]<alpha,na.rm=TRUE)
  size_mat[3,q]=mean(sim500c[,z]<alpha,na.rm=TRUE)
  size_mat[4,q]=mean(sim1000c[,z]<alpha,na.rm=TRUE)
}

pdf(paste("exp3_3.pdf",sep=""))
par(mfrow=c(2,2))
plot(ssize,bias_mat[1:4,2],ylim=c(min(bias_mat[1:4,sequence_n])-0.01,max(bias_mat[1:4,sequence_n])+0.01),
     pch=16,ylab="Bias",xlab="n",col=1)
points(ssize,bias_mat[1:4,4],pch=1,col="slategrey")
points(ssize,bias_mat[1:4,6],pch=2,col="slategrey")
points(ssize,bias_mat[1:4,8],pch=3,col=1)
lines(ssize,bias_mat[1:4,2],lty=3,col=1)
lines(ssize,bias_mat[1:4,4],lty=3,col="slategrey")
lines(ssize,bias_mat[1:4,6],lty=3,col="slategrey")
lines(ssize,bias_mat[1:4,8],lty=3,col=1)
lines(ssize,rep(0,4),lty=2,lwd=0.7,col="darkgray")
ey<-expression(n^{1/2}%*%Bias)
plot(ssize,sqrtn_bias_mat[1:4,2],ylim=c(min(sqrtn_bias_mat[1:4,sequence_n])-0.2,max(sqrtn_bias_mat[1:4,sequence_n])+0.2),
     pch=16,col=1,ylab=ey,xlab="n")
points(ssize,sqrtn_bias_mat[1:4,4],pch=1,col="slategrey")
points(ssize,sqrtn_bias_mat[1:4,6],pch=2,col="slategrey")
points(ssize,sqrtn_bias_mat[1:4,8],pch=3,col=1)
lines(ssize,sqrtn_bias_mat[1:4,2],lty=3,col=1)
lines(ssize,sqrtn_bias_mat[1:4,4],lty=3,col="slategrey")
lines(ssize,sqrtn_bias_mat[1:4,6],lty=3,col="slategrey")
lines(ssize,sqrtn_bias_mat[1:4,8],lty=3,col=1)
lines(ssize,rep(0,4),lty=2,lwd=0.7,col="darkgrey")
#axis(side=1,at=ssize,labels=T)
plot(ssize,size_mat[1:4,2],ylim=c(0,max(size_mat[1:4,sequence_n])+0.1),
     pch=16,col=1,ylab="Size",xlab="n")
points(ssize,size_mat[1:4,4],pch=1,col="slategrey")
points(ssize,size_mat[1:4,6],pch=2,col="slategrey")
points(ssize,size_mat[1:4,8],pch=3,col=1)
lines(ssize,size_mat[1:4,2],lty=3,col=1)
lines(ssize,size_mat[1:4,4],lty=3,col="slategrey")
lines(ssize,size_mat[1:4,6],lty=3,col="slategrey")
lines(ssize,size_mat[1:4,8],lty=3,col=1)
lines(ssize,rep(0.05,4),lty=2,lwd=0.7,col="darkgray")
ey2<-"MC var/est var"
plot(ssize,var_mat[1:4,2],ylim=c(min(var_mat[1:4,sequence_n])-0.2,max(var_mat[1:4,sequence_n])+0.2),
     pch=16,col=1,ylab=ey2,xlab="n")
points(ssize,var_mat[1:4,4],pch=1,col="slategrey")
points(ssize,var_mat[1:4,6],pch=2,col="slategrey")
points(ssize,var_mat[1:4,8],pch=3,col=1)
lines(ssize,var_mat[1:4,2],lty=3,col=1)
lines(ssize,var_mat[1:4,4],lty=3,col="slategrey")
lines(ssize,var_mat[1:4,6],lty=3,col="slategrey")
lines(ssize,var_mat[1:4,8],lty=3,col=1)
lines(ssize,rep(1,4),lty=2,lwd=0.7,col="darkgray")
dev.off()


