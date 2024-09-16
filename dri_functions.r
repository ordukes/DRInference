lambda_select<-function(cov,y){
  model_out<-cv.glmnet(cov,y,alpha=1,family="gaussian",nfold=10)
  lambda=model_out$lambda.min
  return(lambda)
}

lambda_select_binom<-function(cov,y){
  model_out<-cv.glmnet(cov,y,alpha=1,family="binomial",nfold=10)
  lambda=model_out$lambda.min
  return(lambda)
}

bw_select<-function(a,y,G,M){
  V=a-G
  bw_G=npregbw(formula=V~M)$bw
  Z=y-M
  bw_M=npregbw(formula=Z~G)$bw
  list<-list(bw_G=bw_G,bw_M=bw_M)
  return(unlist(list))
}

plug_in.test<-function(N,y,a,M){
  u<-a*(y-M)
  v_u<-var(u)
  chi_stat<-mean(u)^2/(v_u/N)
  p_val<-pchisq(chi_stat,1, lower.tail=FALSE)
  list<-list(est=mean(u)*sqrt(N),se=sqrt(v_u/N),p_val=p_val)
  return(unlist(list))
}

g_estimation.test<-function(N,y,a,G){
  u<-(a-G)*y
  v_u<-var(u)
  chi_stat<-mean(u)^2/(v_u/N)
  p_val<-pchisq(chi_stat,1, lower.tail=FALSE)
  list<-list(est=mean(u)*sqrt(N),se=sqrt(v_u/N),p_val=p_val)
  return(unlist(list))
}

dml_cross.test<-function(N,y,a,G,M){
  u<-(a-G)*(y-M)
  v_u<-var(u)
  chi_stat<-mean(u)^2/(v_u/N)
  p_val<-pchisq(chi_stat,1, lower.tail=FALSE)
  list<-list(est=mean(u)*sqrt(N),se=sqrt(v_u/N),p_val=p_val)
  return(unlist(list))
}

rdml_cross.test<-function(N,y,a,G,M,nsplits,s){
  V <- Z <- Ms <- Gs <- Gnew <- Mnew <- u <- rep(NA,N)          
  for (vfold in 1:nsplits){
    train <- s!=vfold; test <- s==vfold
    if (nsplits==1){ train <- test }
    V[test]=a[test]-G[test]
    if(length(unique(M[test]))==1){
      Gs[test]=predict(lm(V[test]~-1+(M[test]))) 
    } else { 
      bw_G=npregbw(formula=V[test]~M[test])
      Gs[test]=fitted(npreg(bws = bw_G))
    }
    Z[test]=y[test]-M[test]
    if(length(unique(G[test]))==1){
      Ms[test]=predict(lm(Z[test]~-1+(G[test]))) 
    } else {
      bw_M=npregbw(formula=Z[test]~G[test])
      Ms[test]=fitted(npreg(bws = bw_M))
    }    
    Mnew[test]=lm(I(y[test])~-1+offset(M[test])+Gs[test])$fitted
    Gnew[test]=lm(a[test]~-1+offset(G[test])+Ms[test])$fitted
    u[test]=(a[test]-Gnew[test])*(y[test]-Mnew[test])-Gs[test]*(y[test]-Mnew[test])-Ms[test]*(a[test]-Gnew[test])
  }
  v_u<-var(u)
  chi_stat<-mean(u)^2/(v_u/N)
  p_val<-pchisq(chi_stat,1, lower.tail=FALSE)
  list<-list(est=mean(u)*sqrt(N),se=sqrt(v_u/N),p_val=p_val)
  return(unlist(list))
}

dml_cross.est<-function(N,y,a,G,M){
  u<-(a-G)*(y-M)
  den<-(a-G)*a
  list<-list(est=mean(u)/mean(den))
  return(unlist(list))
}

rdml_cross.est<-function(N,y,a,G,M,nsplits,s){
  V <- Z <- Ms <- Gs <- Gnew <- Mnew <- u <- rep(NA,N)          
  for (vfold in 1:nsplits){
    train <- s!=vfold; test <- s==vfold
    if (nsplits==1){ train <- test }
    V[test]=a[test]-G[test]
    if(length(unique(M[test]))==1){
      Gs[test]=predict(lm(V[test]~-1+(M[test]))) 
    } else { 
      bw_G=npregbw(formula=V[test]~M[test])
      Gs[test]=fitted(npreg(bws = bw_G))
    }
    Z[test]=y[test]-M[test]
    if(length(unique(G[test]))==1){
      Ms[test]=predict(lm(Z[test]~-1+(G[test]))) 
    } else {
      bw_M=npregbw(formula=Z[test]~G[test])
      Ms[test]=fitted(npreg(bws = bw_M))
    }    
    Mnew[test]=lm(I(y[test])~-1+offset(M[test])+Gs[test])$fitted
    Gnew[test]=lm(a[test]~-1+offset(G[test])+Ms[test])$fitted
    u[test]=(a[test]-Gnew[test])*(y[test]-Mnew[test])
    den=(a[test]-Gnew[test])*a[test]-Ms[test]*a[test]
  }
  list<-list(est=mean(u)/mean(den))
  return(unlist(list))
}

