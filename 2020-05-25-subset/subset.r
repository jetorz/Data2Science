df = read.csv('skills.csv')
a = df[1]
b = df[1:3,]
a$skills = as.character(a$skills)
a$skills[1] = 'python'