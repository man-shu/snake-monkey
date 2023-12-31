######################################################

# Snake-Monkey Project 

# Author: Rohit (negirohit1997@gmail.com)

# This R script models a response "passing distance" as function of different snake models. 

######################################################

setwd("C:/Users/user/Desktop/glmm")

install.packages("glmmTMB")
library(glmmTMB)

install.packages("DHARMa")
library(DHARMa)

source("diagnostic_fcns.r")

xdata = read.table(file="pd.csv", header=T, sep=",", stringsAsFactor=T)

str(xdata)

# Data checks and preparations
hist(xdata$pd)
table(xdata$pd)
table(xdata$order, xdata$model)

# Removing outliers. (anything above 400)
xdata <- xdata[xdata$pd <= 400, ]

# z-transforming pd
xdata$z.pd = as.vector(scale(xdata$pd))

hist(xdata$z.pd)


## Manually dummy coding  
dummy2 = as.numeric(xdata$model==levels(xdata$model)[2])
dummy3 = as.numeric(xdata$model==levels(xdata$model)[3])

## Centre the dummy coded factors
dummy2= dummy2-mean(dummy2)
dummy3= dummy3-mean(dummy3)

## Model equation

full = glmmTMB(z.pd ~ model + sex +
                   (1 + (dummy2+dummy3)|subj),
                 family=gaussian,
                 data=xdata)

null = glmmTMB(z.pd ~ sex +
                 (1 + (dummy2+dummy3)||subj),
               family=gaussian,
               data=xdata)

  
## full vs null comparrison 
round(as.data.frame(anova(null, full, test="Chisq")), 3)


## Check for overdispersion
overdisp.test(full)

##Calculating scaled residuals
testDispersion(full)

##Calculating Randomized Quantile residuals
simulationOutput <- simulateResiduals(fittedModel = full, n = 1000, plot = T)

## Summary
summary(full)


## Re-level factor, model: CV
xdata$model <- relevel(xdata$model, ref="cv")

## Re-level factor, model: PY
xdata$model <- relevel(xdata$model, ref="py")
