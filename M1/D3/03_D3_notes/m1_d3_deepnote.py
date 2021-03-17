# if 2 + 2 == 4:
#     print(True)

# Function
def how_r_u(x, r=", thanks"):
    return x+r


print("How are you doing?")
how_r_u("okay", r)



# List and Dictionaries

l = ['Gabriel', 'Pathi', "nurlan"]
d = {'Gabriel': 1, 'camelia': 2, 'abishek': 2, 'sven': 2, 'Umidjan':0}
d['Gabriel']

for key in l:
    if key in d:
        d[key]+=1
    else:
        d[key]=1

d