# # Solve 4*ABCD == DCBA
#
# for a in range (1,10):
#     for b in range (0, 10):
#         for c in range (0,10):
#             for d in range (1,10):
#                 first_num = int(str(a)+str(b)+str(c)+str(d))
#                 second_num =  int(str(d)+str(c)+str(b)+str(a))
#                 if 4*first_num == second_num:
#                     print('A =', a)
#                     print('B =', b)
#                     print('C =', c)
#                     print('D =', d)



# ----------------

#
# #5.6
#
# # To take input from the user
# num = int(input("Enter a number: "))
#
# # define a flag variable
# flag = False
#
# # prime numbers are greater than 1
# if num > 1:
#     # check for factors
#     for i in range(2, num):
#         if (num % i) == 0:
#             # if factor is found, set flag to True
#             flag = True
#             # break out of loop
#             break
#
# # check if flag is True
# if flag:
#     print(num, "is not a prime number")
# else:
#     print(num, "is a prime number")
#
#
# #https://www.programiz.com/python-programming/examples/prime-number
#




# ----------------



# #5.7
# import math
#
# a = int(input('Enter first number: '))
# b = int(input('Enter second number: '))
#
# result = math.sqrt(a ** b)
# print(round(result, 2))




# ----------------
# isalpha

string = "Hello, World! How are you!!!"
# string.split()
# string.lower()
string.isalpha()
[char for char in string if char.isalpha()]

for char in string:
    if char.isalpha():
        print(char)



