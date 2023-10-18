import numpy as np
import matplotlib.pyplot as plt
import sympy as sy
from numpy import polyfit
from scipy.stats import linregress
from scipy import interpolate

filename = 'dados.txt'
data = np.loadtxt(filename, skiprows=1, delimiter=' ', dtype=np.int32)
x = data[:, 0]
y = data[:, 1]

#REGRESSAO LINEAR
a, b, R, _, _ = linregress(x,y)
print(f'Regressao linear executada com a = {a:.3f}, b = {b:.3f}')

# ajuste
y2 = a*x + b
#funcao lambda ajuste
y2Lamb = lambda x : a*x + b
print("Regressao Linear para descobrir os valores: ")
print("  2020: ", y2Lamb(2020))
print("  2010: ", y2Lamb(2010))


# figura
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(x,y2,'r--',alpha=0.6,label='modelo')
ax.scatter(x,y,label='alunos/ano enem')
ax.legend()
ax.set_xlabel('ano')
ax.set_ylabel('alunos')
ax.annotate(f'y = {b:.2f} - {-a:.2f}x',
            (2180,10),
            fontsize=12,
            c='r')


#POLYFIT
p2 = np.polyfit(x,y,2)
p3 = np.polyfit(x,y,3) 
p4 = np.polyfit(x,y,4) 

C22 = p2[0]*x**2 + p2[1]*x    + p2[2] # modelo quadrático

# plotagem
fig, ax = plt.subplots(constrained_layout=True)
ax.grid(True)

ax.plot(x,y,  'ko',ms=6, mfc='black', label='dado') 
ax.plot(x,C22,'rs',ms=8, mfc='None', label='Ord2') 

ax.set_xlabel('ano')
ax.set_ylabel('alunos')
ax.legend(loc='best')

C22lambda = lambda x : p2[0]*x**2 + p2[1]*x    + p2[2]
C23lambda = lambda x : p3[0]*x**3 + p3[1]*x**2 + p3[2]*x    + p3[3]
C24lambda = lambda x : p4[0]*x**4 + p4[1]*x**3 + p4[2]*x**2 + p4[3]*x    + p4[4]

print("Polyfit para descobrir os valores(grau 2): ")
print("  2020: ", C22lambda(2020))
print("  2010: ", C22lambda(2010))

print("Polyfit para descobrir os valores(grau 3): ")
print("  2020: ", C23lambda(2020))
print("  2010: ", C23lambda(2010))

print("Polyfit para descobrir os valores(grau 4): ")
print("  2020: ", C24lambda(2020))
print("  2010: ", C24lambda(2010))


#INTERPOLAÇÃO
fig, ax = plt.subplots(constrained_layout=True)
ax.grid(True)


f = interpolate.interp1d(x,y)
t = np.arange(2009, 2023, 1)
m = ['nearest', 'zero', 'slinear', 'quadratic', 'cubic']
F = [interpolate.interp1d(x,y,kind=k) for k in m]
him = [f(t) for f in F]

# plotagem dos valores tabelados
plt.plot(x,y,'*',label='tabelado')
for i in range(5):
    plt.plot(t,him[i],label=m[i])

plt.legend()

# calcula f(2020) para os métodos 'zero', 'slinear' e 'quadratic'
alunos2020 = [f(2020) for f in F[0:]]

print("Interpolate para descobrir os alunos de 2020 com \nmetodos 'nearest', 'zero', 'slinear','quadratic' e 'cubic'")
print("  2020:", alunos2020)


alunos2010 = [f(2015) for f in F[0:]]
print("Interpolate para descobrir os alunos de 2010 com \nmetodos 'nearest', 'zero', 'slinear','quadratic' e 'cubic'")
print("  2010:", alunos2010)



plt.legend();    
plt.show()