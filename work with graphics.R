install.packages("ggplot2")
library(ggplot2)

## 1. 几种常见的基础图
##(1) 散点图 scatter plot

#基础图
plot(mtcars$wt,mtcars$mpg)
#ggplot2  layer的概念

ggplot(mtcars,aes(x=wt,y=mpg)) + geom_point()

##(2) 折线图
head(pressure)

plot(pressure$temperature,pressure$pressure,type="l")
points(pressure$temperature,pressure$pressure)

lines(pressure$temperature, pressure$pressure/2, col="red")
points(pressure$temperature, pressure$pressure/2, col="red")

## ggplot2
library(ggplot2)
ggplot(pressure, aes(x=temperature, y=pressure)) + geom_line()

ggplot(pressure, aes(x=temperature, y=pressure)) + geom_line() + geom_point()

ggplot(pressure, aes(x=temperature, y=pressure)) + geom_line(col="red") + geom_point(col="red")



## (3)条线图


table(mtcars$cyl)
barplot(table(mtcars$cyl))


## ggplot2

ggplot(mtcars, aes(x=factor(cyl))) + geom_bar()


##（4）柱状图

hist(mtcars$mpg)

hist(mtcars$mpg,breaks=10)

qplot(mpg, data=mtcars, binwidth=4)

ggplot(mtcars, aes(x=mpg)) + geom_histogram(bins = 10,fill="white", colour="black")


## (5) box plot
head(ToothGrowth)

boxplot(len ~ supp, data = ToothGrowth)

ggplot(ToothGrowth, aes(x=supp, y=len)) + geom_boxplot()





##### 2. 编辑坐标轴
install.packages("gcookbook")
library(gcookbook)


#(1)  range
head(marathon)
p=ggplot(marathon, aes(x=Half,y=Full)) + geom_point()
P

ggplot(marathon, aes(x=Half,y=Full)) + geom_point() + ylim(80, 600) +xlim(50,200)

p +scale_y_continuous(breaks=seq(100, 420, 40)) + scale_x_continuous(breaks=seq(70, 200, 20))



#(2) flip
head(PlantGrowth)
ggplot(PlantGrowth, aes(x=group, y=weight)) + geom_boxplot()

ggplot(PlantGrowth, aes(x=group, y=weight)) + geom_boxplot() + coord_flip() +
  scale_x_discrete(limits=rev(levels(PlantGrowth$group)))

#(3) change order

ggplot(PlantGrowth, aes(x=group, y=weight)) + geom_boxplot() + scale_x_discrete(limits=c("trt1","trt2","ctrl"))


#(4) date format


str(economics)
econ <- subset(economics, date >= as.Date("1992-06-01") &
                 date < as.Date("1993-06-01"))


p <- ggplot(econ, aes(x=date, y=unemploy)) + geom_line()
p

datebreaks <- seq(as.Date("1992-06-01"), as.Date("1993-06-01"), by="2 month")

install.packages("scales")
library(scales)
p + scale_x_date(breaks=datebreaks, labels=date_format("%Y-%m-%d")) +
  theme(axis.text.x = element_text(angle=90, hjust=1))


#(5) title 

library(gcookbook) # For the data set
head(heightweight)

p <- ggplot(heightweight, aes(x=ageYear, y=heightIn)) + geom_point()

p=p + ggtitle("Age and Height \n\nof School children")
p
p+ theme(plot.title=element_text(size=16, lineheight=.9,face="bold.italic", colour="red"))

p+theme(axis.title.x=element_text(size=16, lineheight=.9,face="bold.italic", colour="red"))




##### 3. color

ggplot(mtcars, aes(x=wt, y=mpg)) + geom_point(aes(colour=cyl))
ggplot(mtcars, aes(x=wt, y=mpg, colour=factor(cyl))) + geom_point()

library(gcookbook) # For the data set
# Base plot
head(uspopage)
p <- ggplot(uspopage, aes(x=Year, y=Thousands, fill=AgeGroup)) + geom_area()
# These three have the same effect
p


# ColorBrewer palette  default bule
p + scale_fill_brewer()

USpop=p + scale_fill_brewer(palette="Oranges") 




# Basic scatter plot
h <- ggplot(heightweight, aes(x=ageYear, y=heightIn, colour=sex)) +
geom_point()
# Default lightness = 65
h
# Slightly darker
h + scale_colour_hue(l=20)


# Base plot
p <- ggplot(uspopage, aes(x=Year, y=Thousands, fill=AgeGroup)) + geom_area()
# The palette with grey:
cb_palette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442",
"#0072B2", "#D55E00", "#CC79A7")
# Add it to the plot
p + scale_fill_manual(values=cb_palette)






###  4. 图片输出
ggplot(uspopage, aes(x=Year, y=Thousands, fill=AgeGroup)) + geom_area() + scale_fill_brewer(palette="Oranges") 
ggsave("C:\\R Data\\myplot.pdf", width=8, height=8, units="cm")



pdf("C:\\R Data\\USpop.pdf", width=3, height=3)
ggplot(uspopage, aes(x=Year, y=Thousands, fill=AgeGroup)) + geom_area() + scale_fill_brewer(palette="Oranges") 

dev.off()


png("C:\\R Data\\USpop.png", width=800, height=800)
ggplot(uspopage, aes(x=Year, y=Thousands, fill=AgeGroup)) + geom_area() + scale_fill_brewer(palette="Oranges") 
dev.off()














### 其他作图包
















