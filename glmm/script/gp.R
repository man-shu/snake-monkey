######################################################

# Snake-Monkey Project 

# Author: Rohit (negirohit1997@gmail.com)

# This R script models a response "gaze percentage" as function of different snake models. 

######################################################

setwd("C:/Users/user/Desktop/glmm")

install.packages("glmmTMB")
library(glmmTMB)

install.packages("DHARMa")
library(DHARMa)

source("diagnostic_fcns.r")

xdata = read.table(file="gp.csv", header=T, sep=",", stringsAsFactor=T)

str(xdata)

# Data checks and preparations
hist(xdata$gp)
table(xdata$gp)
table(xdata$order, xdata$model)

# We are using beta-family and for it the values should lie between 0 to 1 so divide all values by 100 for gp
xdata$gp <- xdata$gp / 100


# creating another data-set for the analysis of the effect of presentation order
order_count <- table(xdata$order)

valid_order <- names(order_count[order_count>2])

xdata.order <- xdata [xdata$order %in% valid_order, ]

table(xdata.order$order)


## Manually dummy coding for Model type for order
dummy2 = as.numeric(xdata.order$model==levels(xdata.order$model)[2])
dummy3 = as.numeric(xdata.order$model==levels(xdata.order$model)[3])


## Manually dummy coding for Model type 
dummy2 = as.numeric(xdata$model==levels(xdata$model)[2])
dummy3 = as.numeric(xdata$model==levels(xdata$model)[3])


## Centre the dummy coded factors
dummy2= dummy2-mean(dummy2)
dummy3= dummy3-mean(dummy3)


## Model equation

## Preliminary
full.2 = glmmTMB(gp ~ model*order + sex +
                   (1 + (dummy2+dummy3)|subj),
                 family=beta_family,
                 data=xdata.order)

full.3 = glmmTMB(gp ~ model*order + sex +
                   (1 + (dummy2+dummy3)||subj),
                 family=beta_family,
                 data=xdata.order)

## Order is non-significant so ignoring order from the final analysis 
full.4 = glmmTMB(gp ~ model + sex +
                   (1 + (dummy2+dummy3)|subj),
                 family=beta_family,
                 data=xdata)

full.5 = glmmTMB(gp ~ model + sex +
                   (1 + (dummy2+dummy3)||subj),
                 family=beta_family,
                 data=xdata)

## Final 
full = glmmTMB(gp ~ model + sex +
                 (1 + (dummy2+dummy3)||subj),
               family=beta_family,
               data=xdata)

null = glmmTMB(gp ~ sex +
                 (1 + (dummy2+dummy3)||subj),
               family=beta_family,
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
