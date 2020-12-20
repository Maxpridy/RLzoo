import csv
import matplotlib.pylab as plt

X = []
returns = []

with open("result_11_21_14_16.txt") as f:
    while True:
        line = f.readline()
        if not line:
            break
        line = line[:-2].split(", ")
        print(line)
        x, y = int(line[0]), float(line[1])
        X.append(x)
        returns.append(y)
        

Y = [0 for _ in range(len(returns)-1)]
y_value = 0
for i, e in enumerate(returns[1:]):
    if e == 'nan':
        Y[i] = y_value
    else:
        y_value = y_value * 0.99 + float(e) * 0.01
        Y[i] = y_value

X = list(map(int, X[1:]))


plt.plot(X, Y)	# line 그래프를 그립니다
plt.show()	