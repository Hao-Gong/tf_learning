import matplotlib.pyplot as plt

xs = []
ys = []
name = 'Anna'
state = 'CA'
gender = 'F'
import matplotlib.pyplot as plt
with open('C:\\Users\\Minchao WU\\Desktop\\Kursmaterialien\\data\\names.csv', 'r') as file:
    print('processing...')
    for line in file:
        line_splitted = line.strip().split(',')
        if line_splitted[1] == name and line_splitted[3] == gender and line_splitted[4] == state:
            xs.append(line_splitted[2])
            ys.append(line_splitted[5])


plt.plot(xs, ys)
plt.yscale('linear')
plt.ylim(50, 800)
plt.show()

print('finish!')