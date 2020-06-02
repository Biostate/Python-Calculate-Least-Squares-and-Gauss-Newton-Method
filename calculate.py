import math
import matplotlib.pyplot as plt
import numpy as np

from numpy.linalg import inv
from  math import e

# Main Function
def ym(t,b):
    return b[0] - pow(e,-b[1]*t)*b[2]*math.sin(b[3]*t+b[4])

# Partial Derivative 1
def ymdb0(t,b):
    return 1

# Partial Derivative 2
def ymdb1(t,b):
    return pow(e,-b[1]*t)*t*b[2]*math.sin(b[3]*t+b[4])

# Partial Derivative 3
def ymdb2(t,b):
    return -pow(e,-b[1]*t)*math.sin(b[3]*t+b[4])

# Partial Derivative 4
def ymdb3(t,b):
    return -pow(e,-b[1]*t)*t*b[2]*math.cos(b[3]*t+b[4])

# Partial Derivative 5
def ymdb4(t,b):
    return -pow(e,-b[1]*t)*b[2]*math.cos(b[3]*t+b[4])

# Main Function
def yp(t):
    return 2*pow(t,5)+pow(t, 4)-2*t-1

def CurveFittingwithLeastSquaresMethodPolynomial(x,y):
    A = []
    for i in x:
        A.append([pow(i,5),pow(i,4),pow(i,3),pow(i,2),i,1])
    A = np.array(A)
    transpose = A.transpose()
    Left = transpose.dot(A)
    Right = transpose.dot(y)
    result = np.linalg.inv(Left).dot(Right)
    fit = [None] * 50
    for i in range(len(A)):
        fit[i] = result[0] *pow(x[i],5) + result[1] * pow(x[i],4) + result[2] * pow(x[i],3) + result[3] * pow(x[i],2) + result[4] * x[i] + result[5]
    
    return fit,result

def CurveFittingwithLeastSquaresMethod(x,y):
    x= np.array([x,[1]*len(x)])
    transpose = np.array(x).transpose()
    multA = x.dot(transpose)
    multB = x.dot(y)
    result = np.linalg.inv(multA).dot(multB)
    fit = [None] * 50
    for i in range(len(x[0])):
        fit[i] = result[0]*x[0][i]+result[1]
    return fit,result

def CalculateLSQLIN():
    print("Bilgilendirme: Verilerinizi x,y satır formatında yazdığınızdan emin olunuz.")
    filename = input("Lütfen okumak istediğiniz dosyanın adını yazın:")
    f = open(filename, "r")
    x = []
    y = []
    for line in f:
        line = line.rstrip().split(',')
        x.append(float(line[0]))
        y.append(float(line[1]))
    y = np.array(y)
    f.close()
    fig, axs = plt.subplots()
    axs.scatter(x, y)
    Lineer, LValues = CurveFittingwithLeastSquaresMethod(x,y)
    axs.plot(x, Lineer, color='green', linewidth=1, markersize=0, linestyle="-.")
    Polynomial,Values = CurveFittingwithLeastSquaresMethodPolynomial(x,y)
    axs.plot(x, Polynomial, linewidth=2, markersize=0, color='orange')
    fig.suptitle('En Küçük Kareler Metodu İle Polinomial Eğri Uydurma')
    plt.legend(['Doğru Eğrisi', 'Polinomial Eğri', 'Veri'], loc=4)
    print('En Küçük Kareler Metodu İle Polinomial Eğri Uydurma Sonuçları:');
    print("a0="+str(Values[5])+", a1="+str(Values[4])+", a2="+str(Values[3])+", a3="+str(Values[2])+", a4="+str(Values[1])+", a5="+str(Values[0]))
    print('En Küçük Kareler Metodu İle Polinomial Eğri Uydurma Hata Sonuçları:');
    plt.show()
    choise = input("Bu değerleri kaydetmek istiyor musunuz? (E, H):")
    if(choise.lower() == "e"):
        SaveName = input("Lütfen kaydetmek istediğiniz ismi dosya formatı olmadan yazın: ")
        saveFile = open(SaveName+".txt", "w")
        saveFile.write("a0="+str(Values[5])+", a1="+str(Values[4])+", a2="+str(Values[3])+", a3="+str(Values[2])+", a4="+str(Values[1])+", a5="+str(Values[0])+'\n')
        saveFile.close()
        print(SaveName+'.txt kaydedildi:');

def GaussNewtonMethod(startpoints):
    x = []
    y = []
    print("Bilgilendirme: Verilerinizi x,y satır formatında yazdığınızdan emin olunuz.")
    filename = input("Lütfen okumak istediğiniz dosyanın adını yazın:")
    f = open(filename, "r")
    for line in f:
        line = line.rstrip().split(',')
        x.append(float(line[0]))
        y.append(float(line[1]))
    y = np.array(y)
    CalculatedPoints = startpoints
    for k in range(5):
        # Calculate Z
        Z = []
        for i in range(len(x)):
            Z.append([
                      ymdb0(x[i],CalculatedPoints),
                      ymdb1(x[i],CalculatedPoints),
                      ymdb2(x[i],CalculatedPoints),
                      ymdb3(x[i],CalculatedPoints),
                      ymdb4(x[i],CalculatedPoints)
                      ])
        D = []
        for i in range(len(y)):
            D.append(y[i]-Z[i][0]);
        D = np.array(D)
        Z = np.array(Z)
        ZT = Z.transpose()
        eq1 = ZT.dot(Z)
        eq2 = inv(eq1)
        eq3 = eq2.dot(ZT)
        eq4 = eq3.dot(D)
        CalculatedPoints = np.add(startpoints,eq4)
        
    newy = []
    print('B0: ',CalculatedPoints[0],'B1: ',CalculatedPoints[1],'B2: ',CalculatedPoints[2],'B3: ',CalculatedPoints[3],'B4: ',CalculatedPoints[4])
    for i in x:
        #TODO: T and ColculatedPoints
        newy.append(ym(i,CalculatedPoints))
    fig, axs = plt.subplots()
    axs.scatter(x, y)
    axs.plot(x, newy, linewidth=2, markersize=0, color='orange')
    fig.suptitle('Gauss - Newton Eğri Uydurma')
    plt.legend(['Gauss-Newton', 'Veri'], loc=4)
    plt.show()
    choise = input("Bu değerleri kaydetmek istiyor musunuz? (E, H):")
    if(choise.lower() == "e"):
        SaveName = input("Lütfen kaydetmek istediğiniz ismi dosya formatı olmadan yazın: ")
        saveFile = open(SaveName+".txt", "w")
        saveFile.write("B0="+str(CalculatedPoints[0])+", B1="+str(CalculatedPoints[1])+", B2="+str(CalculatedPoints[2])+", B3="+str(CalculatedPoints[3])+", B4="+str(CalculatedPoints[4])+'\n')
        saveFile.close()
        print(SaveName+'.txt kaydedildi:');
    

UserChoise = ""
while UserChoise != "q":
    print("Menü:")
    print("1) En Küçük Kareler Metodu İle Polinomial Eğri Uydurma ")
    print("2) Gauss-Newton Eğri Uydurma ")
    print("Çıkmak için q tuşuna basın")
    UserChoise = input("Lütfen yapmak istediğiniz seçeneğin numarasını girin:")
    if UserChoise == "1":
        CalculateLSQLIN()
    elif UserChoise == "2":
        GaussNewtonMethod([1.8, 0.1389, 1.8031, 2.3529, 1.5118])
        # GaussNewtonMethod([0.55, 0.4545, 0.5532, 4.2397, 1.464])
    
