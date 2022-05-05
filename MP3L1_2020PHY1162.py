import matplotlib.pyplot as plt
import math
import numpy as np
import numba as nb
import pandas as pd
plt.style.use("seaborn-dark-palette")

@nb.vectorize
def exp(x,a,n):
    sum_ = 1
    for i in np.arange(1,n):
        sum_ += (x-a)**(i)/math.gamma(i+1)
    return(sum_)

@nb.vectorize
def MySinSeries(x,a,n):  #returns y as vector b/c of vectorize function
    sum_ = 0
    for i in np.arange(n):
        sum_ += (-1)**i*(x-a)**(2*i+1)/math.gamma(2*i+2)
    return(sum_)

@nb.vectorize
def MyCosSeries(x,a,n):
    sum_ = 0
    for i in np.arange(n):
        sum_ += (-1)**i*(x-a)**(2*i)/math.gamma(2*i+1)
    return(sum_)

'''this function will take input as function of taylor series 
expansion, the vector x and tolerence of y and outputs the no. of terms required in 
series expansion along with the result for given accuracy'''

def Acc_expansion(f,x,a,tol):
    n = 200
    N = []
    func_acc = []
    for k in range(len(x)):
       for i in range(2,n):
           new_f = f(x[k],a,i)
           old_f = f(x[k],a,i-1)
           if new_f == old_f :
               abs1 = 0
           else:
               abs1 = abs(new_f - old_f)/abs(max(new_f,old_f))
           
           if abs1 == 0 :
              n1 =  1
              break         
           elif abs1 <= tol :
              n1 = i
              break
       func_acc.append(new_f)
       N.append(n1)
    return N,func_acc

        
    
xs= np.linspace(-2*np.pi,2*np.pi,50)
m = [1,2,5,10,20]
plttyp = ['--1','--<','--x','--','--']
 
x0 = np.pi/4
m2 = np.arange(2,22,2)
     
# for Sine series    
yj_sin = np.array([MySinSeries(xs,0,i) for i in m],dtype=float)
    
y0_sin = np.array(MySinSeries(x0,0,m2))


'''
fig,(ax1,ax2) = plt.subplots(1,2)
for j in range(len(m)) : 
    ax1.plot(xs,yj_sin[j],plttyp[j],label=f"m={m[j]}")
ax1.plot(xs,np.sin(xs),label = 'Inbuilt sine function')
ax1.set_ylim(-10,10)
ax1.set_xlabel('x')
ax1.set_ylabel('sin(x)')
ax2.plot(m2,y0_sin,'--*',label = 'MySinSeries(\u03C0/4,n)')
#ax2.plot(m2,[np.sin(x0)]*len(m2),label = 'sin(\u03C0/4)')
ax2.set_xlabel('No. of terms')
ax2.set_ylabel('value of sin at \u03C0/4 ')
fig.suptitle('Sin Taylor expansion analysis')
ax2.legend()
ax1.legend()'''
    
# for cos series    
yj_cos = np.array([MyCosSeries(xs,0,i) for i in m],dtype=float)
    
y0_cos = np.array(MyCosSeries(x0,0,m2))
    
'''
y_ner = []
for i in range(1,len(y0_cos)):
    y_ner.append(y0_cos[i]-y0_cos[i-1])

    
    
y_namme = ['Y0(4)-Y0(2)','Y0(6)-Y0(4)','Y0(8)-Y0(6)','Y0(10)-Y0(8)','Y0(12)-Y0(10)','Y0(14)-Y0(12)','Y0(16)-Y0(14)','Y0(18)-Y0(16)','Y0(20)-Y0(18)']    
    
d2 = {'list':y_namme,'Error':y_ner}
df2 = pd.DataFrame(data = d2)
df2.to_csv('table2.csv')'''

'''
fig,(ax1,ax2) = plt.subplots(1,2)
for j in range(len(m)) : 
    ax1.plot(xs,yj_cos[j],plttyp[j],label=f"m={m[j]}")
ax1.plot(xs,np.cos(xs),label = 'Inbuilt Cos function')
ax1.set_ylim(-10,10)
ax1.set_xlabel('x')
ax1.set_ylabel('Cos(x)')
ax2.plot(m2,y0_cos,'--*',label = 'MyCosSeries(\u03C0/4,n)')
ax2.plot(m2,[np.cos(x0)]*len(m2),label = 'cos(\u03C0/4)')
ax2.set_xlabel('No. of terms')
ax2.set_ylabel('value of Cos at \u03C0/4 ')
fig.suptitle('Cos Taylor expansion analysis')
ax2.legend()
ax1.legend()'''


# Question 2 


x_ac = np.arange(0,np.pi + (np.pi/8),np.pi/8) # x in [0 , pi ]

#for a accuracy of n significant digits we require a rel tol of 0.5*10**(-n)

tol1 = 0.5*10**(-3)      #for plotting
tol2 = 0.5*10**(-6)      #for tabulation  

N_6,Sin_ap6 = Acc_expansion(MySinSeries, x_ac, 0, tol2)

sin_ap6tb = ["%.7g" % k for k in Sin_ap6]

sin_tb2 = ["%.7g" % k for k in np.sin(x_ac)]

d1 = {'X':x_ac,'Sin(x)(calc)':sin_ap6tb,'No. of terms':N_6,'Sin(x)(inbuilt)':sin_tb2}
df1 = pd.DataFrame(data=d1)
df1.to_csv('table1.csv')

print('Sin calcualted at these points for accuracy upto 6 significant digits => \n',df1)


N_3 , Sin_ap3 = Acc_expansion(MySinSeries, xs, 0, tol1)
we_sin = np.sin(xs)

print(np.sin(np.pi))
new = MySinSeries(np.pi,0,18)
old = MySinSeries(np.pi,0,17)

print(new)
print(old)
p = 100
rel = abs(new-old)/(((abs(new)**(p) + abs(old)**(p))/2)**(1/p))

print(rel)


'''
fig,ax = plt.subplots()
ax.plot(xs,Sin_ap3,'o',label = 'Calculated sin for accuracy upto 3 significant digits')
ax.plot(xs,we_sin,label = 'Continuous sin curve')
ax.set_title('Sin(x) calculated and analytic')
ax.set_xlabel('x')
ax.set_ylabel('Sin(x)')
plt.legend()'''
    


