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


# creating another data-set for the analysis of the effect of presentation order
order_count <- table(xdata$order)

valid_order <- names(order_count[order_count>2])

xdata.order <- xdata [xdata$order %in% valid_order, ]

table(xdata.order$order)


## Manually dummy coding for Model type 
dummy2 = as.numeric(xdata.order$model==levels(xdata.order$model)[2])
dummy3 = as.numeric(xdata.order$model==levels(xdata.order$model)[3])

## Centre the dummy coded factors
dummy2= dummy2-mean(dummy2)
dummy3= dummy3-mean(dummy3)


## Model equation

## Preliminary
full.2 = glmmTMB(pd ~ model*order + sex +
                   (1 + (dummy2+dummy3)|subj),
                 family=gaussian,
                 data=xdata.order)

full.3 = glmmTMB(pd ~ model*order + sex +
                   (1 + (dummy2+dummy3)||subj),
                 family=gaussian,
                 data=xdata.order)


full.4 = glmmTMB(pd ~ model*order + sex +
                   (1 + (dummy2+dummy3)||subj),
                 family=gaussian(link="log"),
                 data=xdata.order)

## Given order is significant we will have to control for order
full.5 = glmmTMB(pd ~ model + sex +
                 (1 + (dummy2+dummy3)|subj) +
               (1|order),
               family=gaussian,
               data=xdata.order)

## Final - 
full = glmmTMB(pd ~ model + sex +
                   (1 + (dummy2+dummy3)|subj) +
                   (1|order),
                 family=gaussian,
                 data=xdata.order)

null = glmmTMB(pd ~ sex +
                 (1 + (dummy2+dummy3)|subj) +
                 (1|order),
               family=gaussian,
               data=xdata.order)

  
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
xdata.order$model <- relevel(xdata.order$model, ref="cv")

## Re-level factor, model: PY
xdata.order$model <- relevel(xdata.order$model, ref="py")
