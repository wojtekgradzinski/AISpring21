# Counting words line by line.
import os
path = os.getcwd() + '/text-files/blakepoems.txt'
print(path)
fp = open(path, 'r+')
text = fp.readlines()
print(text)
