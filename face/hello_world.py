print("How many times do you want me to say the word 'sun'?")

loop = input("Enter the number of times... ")
loop = int(round(float(loop), 0))

if (loop > 10000):
    print("You're number is too big!!!! ")
    exit()

for i in range(loop):
    print("{}. sun".format(i + 1))