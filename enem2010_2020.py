import pandas as pd
from bokeh.io import output_notebook
from bokeh.plotting import figure, show
from bokeh.models import PrintfTickFormatter, Range1d

from scipy import stats
from scipy import interpolate
import numpy as np

filename = 'dados.txt'
data = np.loadtxt(filename, skiprows=1, delimiter=' ', dtype=np.int32)
x = data[:, 0]
y = data[:, 1]
print(x[0], y[0])
f = figure()
f.scatter(x,y,fill_color="blue",radius=0.3,alpha=1)

# formatacao
f.background_fill_color = "blue"
f.background_fill_alpha = 0.02
f.x_range=Range1d(2000, 2025)
e = 5.e5
f.y_range=Range1d(y.min() - e,y.max() + e)
f.xaxis.axis_label = "Ano"
f.yaxis.axis_label = "Alunos"
f.yaxis[0].formatter = PrintfTickFormatter(format="%1.2e")
f.line(x,y,color='green')
p = interpolate.interp1d(x, y)
print(p)
xx = [2010, 2020]
yy = p(xx)
print("Quantidade de Alunos estimada que fez o ENEM em: 2010 = {:d}; 2020 = {:d}; ".format(int(yy[0]),int(yy[1])))
f.square(xx,yy,color='black',line_width=4)
show(f)