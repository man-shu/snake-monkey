######################################################

# Snake-Monkey Project 

# Author: Rohit (negirohit1997@gmail.com)

# This R script models a response, closest proximity acquired, as function of different snake models. 

######################################################

setwd("C:/Users/user/Desktop/glmm")

install.packages("glmmTMB")
library(glmmTMB)

install.packages("DHARMa")
library(DHARMa)

source("diagnostic_fcns.r")

xdata = read.table(file="pr.csv", header=T, sep=",", stringsAsFactor=T)

str(xdata)

# Data checks and preparations
hist(xdata$pr)

# Removing outliers. (anything above 400)
xdata <- xdata[xdata$pr <= 400, ]

# Data inspection 
hist(xdata$pr)

## Manually dummy coding for Model type 
dummy2 = as.numeric(xdata$model==levels(xdata$model)[2])
dummy3 = as.numeric(xdata$model==levels(xdata$model)[3])

## Centre the dummy coded factors
dummy2= dummy2-mean(dummy2)
dummy3= dummy3-mean(dummy3)

## Model equation

## Final 
full = glmmTMB(pr ~ model + sex +
                   (1 + (dummy2+dummy3)||subj),
                 family = nbinom1(link = "log"),
                 data = xdata)

null =  glmmTMB(pr ~ sex +
                  (1 + (dummy2+dummy3)||subj),
                family =  nbinom1(link = "log"),
                data = xdata)


## full vs null comparrison 
round(as.data.frame(anova(null, full, test="Chisq")), 3)


## Assumptions
## Check for overdispersion
overdisp.test(full)

##Calculating scaled residuals using Dharma
testDispersion(full)

##Calculating Randomized Quantile residuals usind Dharma
simulationOutput <- simulateResiduals(fittedModel = full, n = 1000, plot = T)

## Summary
summary(full)


## Re-level factor, model: CV
xdata$model <- relevel(xdata$model, ref="cv")

## Re-level factor, model: PY
xdata$model <- relevel(xdata$model, ref="py")


## Re-level factor, sex: F
xdata$model <- relevel(xdata$sex, ref="F")
