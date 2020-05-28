import pandas as pd

lst = ['a', 1, 'book']
a = lst
a is lst
# True
lst[1] = 100
lst
# ['a', 100, 'book']
a[0:2] = ['a1', 'a2']
lst
# ['a1', 'a2', 'book']
a is lst
# True

df = pd.read_csv('skills.csv', 'r', delimiter=',', encoding='utf8',skipinitialspace=True)
df