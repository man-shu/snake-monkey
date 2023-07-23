######################################################

# Snake-Monkey Project 

# Author: Rohit (negirohit1997@gmail.com)

# This R script models a response "bipedal standing" as function of different snake models. 

######################################################

setwd("C:/Users/user/Desktop/glmm")

install.packages("glmmTMB")
library(glmmTMB)

install.packages("DHARMa")
library(DHARMa)

source("diagnostic_fcns.r")

xdata = read.table(file="bs.csv", header=T, sep=",", stringsAsFactor=T)


str(xdata)

# Data checks and preparations
hist(xdata$bs)
table(xdata$bs)
table(xdata$order, xdata$model)


# Delete 'bb' from the data set because there are no instances of 'bs' for the 'bb' category.
xdata <- subset(xdata, !(model == 'bb'))

# creating another data-set for the analysis of the effect of presentation order
order_count <- table(xdata$order)

valid_order <- names(order_count[order_count>2])

xdata.order <- xdata [xdata$order %in% valid_order, ]

table(xdata.order$order)


## Manually dummy coding for Model type for order
dummy2 = as.numeric(xdata.order$model==levels(xdata.order$model)[2])


## Manually dummy coding for Model type 
dummy2 = as.numeric(xdata$model==levels(xdata$model)[2])


## Centre the dummy coded factors
dummy2= dummy2-mean(dummy2)


## Model equation

## Preliminary
full.2 = glmmTMB(bs ~ model*order + sex +
                   (1 + (dummy2)|subj),
                 family = truncated_poisson(link = "log"),
                 ziformula = ~1,
                 data = xdata.order)

full.3 = glmmTMB(bs ~ model*order + sex +
                   (1 + (dummy2)||subj),
                 family = truncated_poisson(link = "log"),
                 ziformula = ~1,
                 data = xdata.order)


full.4 = glmmTMB(bs ~ model*order + sex +
                   (1 + (dummy2)|subj),
                 family = truncated_nbinom1(link = "log"), 
                 ziformula = ~1,
                 data = xdata.order)

full.5 = glmmTMB(bs ~ model*order + sex +
                   (1 + (dummy2)||subj),
                 family = truncated_nbinom1(link = "log"), 
                 ziformula = ~1,
                 data = xdata.order)

## Order is non-significant so ignoring order from the final analysis 
full.6 = glmmTMB(bs ~ model + sex +
                   (1 + (dummy2)|subj),
                 family = truncated_poisson(link = "log"),
                 ziformula = ~1,
                 data = xdata)


## Final 
full = glmmTMB(bs ~ model + sex +
                 (1 + (dummy2)|subj),
               family = truncated_poisson(link = "log"),
               ziformula = ~1,
               data = xdata)

null = glmmTMB(bs ~ sex +
                 (1 + (dummy2)||subj),
               family = truncated_poisson(link = "log"),
               ziformula = ~1,
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
