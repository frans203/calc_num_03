import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from numpy import polyfit
from scipy.stats import linregress
from scipy import interpolate

#anos perdidos: 1985, 1995 e 2010
filename = 'dados-01.txt'
data = np.loadtxt(filename, skiprows=1, delimiter=' ', dtype=np.int32)
x = data[:, 0]
y = data[:, 1]

#Regressao Linear
a, b, R, _, _ = linregress(x,y)
print(f'Regressao linear executada com a = {a:.3f}, b = {b:.3f}')

y2 = a*x + b
y2Lamb = lambda x : a*x + b
print("Regressao Linear para descobrir os valores: ")
print("  1985: ", y2Lamb(1985))
print("  1995: ", y2Lamb(1995))
print("  2010: ", y2Lamb(2010))


fig, ax = plt.subplots(constrained_layout=True)
ax.plot(x,y2,'r--',alpha=0.6,label='modelo')
ax.scatter(x,y,label='Instituições/ano')
ax.set_xlabel('ano')
ax.set_ylabel('Instituições(Publicas & Privadas)')
ax.annotate(f'y = {b:.2f} - {-a:.2f}x',
            (0,0),
            fontsize=12,
            c='r')
ax.legend()


#polyfit
p2 = np.polyfit(x,y,2)
p3 = np.polyfit(x,y,3) 
p4 = np.polyfit(x,y,4) 

C22 = p2[0]*x**2 + p2[1]*x  + p2[2] # modelo quadrático

fig, ax = plt.subplots(constrained_layout=True)
ax.grid(True)

ax.plot(x,y,  'ko',ms=6, mfc='black', label='dado') 
ax.plot(x,C22,'rs',ms=8, mfc='None', label='Ord2') 

ax.set_xlabel('ano')
ax.set_ylabel('Instituições(Publicas & Privadas)')
ax.legend(loc='best')

C22lambda = lambda x : p2[0]*x**2 + p2[1]*x    + p2[2]
C23lambda = lambda x : p3[0]*x**3 + p3[1]*x**2 + p3[2]*x    + p3[3]
C24lambda = lambda x : p4[0]*x**4 + p4[1]*x**3 + p4[2]*x**2 + p4[3]*x    + p4[4]

print("Polyfit para descobrir os valores(grau 2): ")
print("  1985: ", C22lambda(1985))
print("  1995: ", C22lambda(1995))
print("  2010: ", C22lambda(2010))

print("Polyfit para descobrir os valores(grau 3): ")
print("  1985: ", C23lambda(1985))
print("  1995: ", C23lambda(1995))
print("  2010: ", C23lambda(2010))


print("Polyfit para descobrir os valores(grau 4): ")
print("  1985: ", C24lambda(1985))
print("  1995: ", C24lambda(1995))
print("  2010: ", C24lambda(2010))


# Interpolação 
fig, ax = plt.subplots(constrained_layout=True)
ax.grid(True)


f = interpolate.interp1d(x,y)
t = np.arange(1980, 2025, 5)
m = ['nearest', 'zero', 'slinear', 'quadratic', 'cubic']
F = [interpolate.interp1d(x,y,kind=k) for k in m]
him = [f(t) for f in F]

# plotagem dos valores tabelados
plt.plot(x,y,'*',label='tabelado')
for i in range(5):
    plt.plot(t,him[i],label=m[i])

plt.legend()
# calcula f(1985) para os métodos 'nearest', 'zero', 'slinear' e 'quadratic'
inst1985 = [f(1985) for f in F[0:]]

print("Interpolate para descobrir os alunos de 1985 com \nmetodos 'nearest', 'zero', 'slinear','quadratic' e 'cubic'")
print("  1985:", inst1985)

# calcula f(1995) para os métodos 'nearest', 'zero', 'slinear' e 'quadratic'
inst1995 = [f(1995) for f in F[0:]]

print("Interpolate para descobrir os alunos de 1995 com \nmetodos 'nearest', 'zero', 'slinear','quadratic' e 'cubic'")
print("  1995:", inst1995)


# calcula f(2010) para os métodos 'nearest', 'zero', 'slinear' e 'quadratic'
inst2010 = [f(2010) for f in F[0:]]

print("Interpolate para descobrir os alunos de 2010 com \nmetodos 'nearest', 'zero', 'slinear','quadratic' e 'cubic'")
print("  2010:", inst2010)


plt.show()