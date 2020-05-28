library(ggplot2)
sdata = read.csv('skills.csv')
sdata$skills = reorder(sdata$skills, sdata$numbers)
ggplot(data=sdata) + geom_col(mapping = aes(x=skills, y=numbers)) +
    theme(axis.text.x = element_text(angle = 90)) + coord_flip()

sdata = read.csv('skills.csv')
s = as.character(sdata$skills)
sdata$skills = factor(s, levels = rev(s))
ggplot(data=sdata) + geom_col(mapping = aes(x=skills, y=numbers)) +
    theme(axis.text.x = element_text(angle = 90)) + coord_flip()
