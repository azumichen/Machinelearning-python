library(ggplot2)
library(dplyr)

housing = read.csv("Q:\\培训\\EDA\\housing.csv",header=T,sep=",")
attach(housing)
head(housing)

str(housing)
dim(housing)
head(housing)


##单变量分析
summary(MedPrice)
ggplot(housing, aes(x=MedPrice)) + geom_histogram(bins = 10,fill="white", colour="black")



summary(CRIM)
ggplot(housing, aes(x=CRIM)) + geom_histogram(bins = 10,fill="white", colour="black")

##0,1
summary(residLand)
ggplot(housing, aes(x=residLand)) + geom_histogram(bins = 10,fill="white", colour="black")


summary(room)
ggplot(housing, aes(x=room)) + geom_histogram(bins = 10,fill="white", colour="black")

##多变量分析

#log transformation
ggplot(housing,aes(CRIM,MedPrice)) + geom_point()

housing=mutate(housing, logCRIM = log(CRIM))

ggplot(housing,aes(logCRIM,MedPrice)) + geom_point()


### continuous variable convert to categorical
ggplot(housing,aes(residLand,MedPrice)) + geom_point()


housing$residLandf = cut(housing$residLandf, breaks = c(0,0.0001,50,Inf), right = FALSE, labels = c("0","<50" ,">50"))

ggplot(housing, aes (residLandf,MedPrice,color=residLandf)) + 
  geom_boxplot(aes(fill=residLandf), alpha=0.3)


### detect outlier

ggplot(housing,aes(room,MedPrice)) + geom_point()

housing[which(room>60),]
housing=housing[-which(room>60),]

ggplot(housing,aes(room,MedPrice)) + geom_point()


### stu/teacher ration
ggplot(housing,aes(teacher,MedPrice)) + geom_point()

housing$teacherF = cut(housing$teacher, breaks = c(0,15,17.5,20,Inf), right = FALSE, labels = c("<15","15-17.5" ,"17.5-20",">20"))

ggplot(housing, aes(factor(teacherF),MedPrice,color=factor(teacherF))) + 
  geom_boxplot(aes(fill=factor(teacherF)), alpha=0.3) 



### factor boxplot

ggplot(housing, aes(factor(highwayAccess),MedPrice,color=factor(highwayAccess))) + 
  geom_boxplot(aes(fill=factor(highwayAccess)), alpha=0.3) 


ggplot(housing, aes(factor(River),MedPrice,color=factor(River))) + 
  geom_boxplot(aes(fill=factor(River)), alpha=0.3) 


##
ggplot(housing,aes(LSTAT,MedPrice)) + geom_point(aes(color = highwayAccess))





## regression



housing.lm = lm(MedPrice~ logCRIM+ residLandf +teacherF+TAX + buzLand +River +room+houseAge+highwayAccess
                +LSTAT, data =housing)

housing.lm2 = lm(MedPrice~  logCRIM +teacherF+TAX + buzLand +River +room+houseAge+highwayAccess
                +LSTAT, data =housing)

housing.lm3 = lm(MedPrice~  logCRIM +teacherF+TAX + buzLand +River +room+highwayAccess
                 +LSTAT, data =housing)

housing.lm4 = lm(MedPrice~  logCRIM +teacherF+TAX  +River +room+highwayAccess
                 +LSTAT, data =housing)

housing.lm5 = lm(MedPrice~  logCRIM +teacherF+TAX  +River +room+LSTAT, data =housing)

housing.lm6 = lm(MedPrice~   teacherF+TAX  +River +room+LSTAT, data =housing)

housing.lm7 = lm(MedPrice~   teacherF  +River +room+LSTAT, data =housing)


# Zone_id CRIM residLand buzLand River  room houseAge  disBC highwayAccess TAX teacher LSTAT      MedPrice   logCRIM residLandf teacherF
#  75      76 0.09512         0   12.83     0 6.286       45 4.5026             5 398    18.7  8.94     21.4  -2.352616          0  17.5-20


summary(housing.lm7)

new = data.frame(teacherF="17.5-20",River=0,room=6.286,LSTAT=8.94)

predict.lm(housing.lm7,new,interval= "prediction")






ggplot(housing,aes(TAX,MedPrice)) + geom_point()

housing$TAXF = cut(housing$TAX, breaks = c(0,250,350,500,Inf), right = FALSE, labels = c("<250","250-350" ,"350-500",">500"))

ggplot(housing, aes(factor(TAXF),MedPrice,color=factor(TAXF))) + 
  geom_boxplot(aes(fill=factor(TAXF)), alpha=0.3) 


##

summary(housing)


##
pairs(housing)

##


pairs(automobile)




library(ggplot2)### 

ggplot(housing, aes(factor(highwayAccess),MedPrice,color=factor(highwayAccess))) + 
  geom_boxplot(aes(fill=factor(highwayAccess)), alpha=0.3) +
  xlab("Risk level") +
  theme(legend.position = "none")





par(mfrow = c(2, 2)
    