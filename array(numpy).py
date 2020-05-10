#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
a=np.array([[1,2],
            [3,4]])
b=np.array([[4,3],
            [2,1]])
#adding 1 to every element...
print("Adding 1 to every element\n",a+1)
#subtract 2 to every element....
print("Sutract to 2 every element\n",b-2)
#sum of array elements...
print("Sum of array is",a.sum())
#adding two arrays...
print("Adding two arrays is :",a+b)


# # Advance NUMPY
# 

# In[4]:


import numpy as np
a1=np.array([[1,2,3],
             [4,5,6]])
b1=np.array([[7,8,9],
             [10,11,12]])
print("\t\t\tStacking")
#np.vstack:To stack array along vertical axis...
print("\nVertical stacking :\n",np.vstack((a1,b1)))
#np.hstack: To stack array along horizontal axis...
print("\nHorizontal stacking:\n",np.hstack((a1,b1)))
#np.coloumn_Stack:TO stack 1D arrays as coloumn into 2D array...
c=[5,6]
print("\ncolumn stacking\n",np.column_stack((a1,c)))
#np.concatenate:To stack arrays along specified axis
print("\n concatetenating to the 2nd axis\n",np.concatenate((a1,b1),axis=1))


# In[22]:


import numpy as np
a2=np.array([[1,3,5,7,9,11],
             [2,4,6,8,10,12]])
print("\t\t\tSplitting")
#np.hsplit : split array along horizontal axis...
print("\nSpliting along horizontal axis into 2 parts :\n",np.hsplit(a2,3))
#np.vsplit : split array along vertical axis...
print("\nSpliting along vertical axis into 2 parts :\n",np.vsplit(a2,2))
#np.array_split: split array along specified axis...
grid=np.arange(16).reshape(4,4)
print(grid)
upper,lower=np.vsplit(grid,[2])
print("Upper array\n",upper)
print("lower array\n",lower)
left,right=np.hsplit(grid,[2])
print("Left side array\n",left)
print("Right side array\n",right)


# # The BroadCasting Rule...

# In[ ]:


x=np.array([1.0,2.0,3.0])
y=2.0
print(x*y)
z=[2.0,2.0,2.0]
print(x*z)


# In[ ]:


a3=np.array([0.0,10.0,20.0,30.0])
b3=np.array([0.0,1.0,2.0])
print(a3[:,np.newaxis]+b3)


# # Working with datetime...

# In[ ]:


import numpy as np
#creating a date...
today=np.datetime64("2017-02-01")
print("Date is :",end="")
print(today)
print("year is :",np.datetime64(today,'Y'))
print("year & Months is :",np.datetime64(today,'M'))
#creating array of date in a month..
dates=np.arange("2017-02","2017-03", dtype="datetime64[D]")
print("\nDates of feb, 2017 is :",dates)
print("\nToday is a feb",today in dates)
#Arthemetic Operations on dates...
dur=np.datetime64("2017-05-22") - np.datetime64("2016-05-22")
print("\nno. of days :",dur)
print("no. of weeks:",np.timedelta64(dur,"W"))
#sorting dates..
s=np.array(["2017-02-12","2107-10-13","2019-05-22"],dtype="datetime64")
print("\nDates in sorted order\n",np.sort(s))


# # Linear alegebra In NUMPY...

# In[ ]:


import numpy as np
A=np.array([[6,1,1],
            [4,-2,5],
            [2,8,7]])
print("Rank A :",end="")
print(np.linalg.matrix_rank(A))
print("\nTrace of A :",end="")
print(np.trace(A))
print("\nDeterminant of A",end="")
print(np.linalg.det(A))
print("\nInverse of A:")
print(np.linalg.inv(A))
print("\n Matrix A raised to the power of 3 :")
print(np.linalg.matrix_power(A,3))


# In[ ]:


import numpy as np
a4=np.array([[1,2,],[3,4]])
b4=np.array([8,18])
print("Solution of linear equation is :",np.linalg.solve(a4,b4))


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
a5=np.arange(0,9)
b5=np.array([a5,np.ones(9)])
a6=[19,20,20.5,21,5,22,23,25.5,24]
w=np.linalg.lstsq(b5.T,a6)[0]
line=w[0]*a5 + w[1]
plt.plot(a5,line, "r-")
plt.plot(a5,a6,"o")
plt.show()


# # Basic Slicing and Adavance indexing in numpy........

# In[ ]:


import numpy as np
list1=[1,2,3,4,5,6,]
list2=[10,9,8,7,6,5]
l1=np.array(list1)
l2=np.array(list2)
print("Multiply List{} * List2{}".format(l1,l2),end="=")
print(l1*l2)


# In[ ]:


import numpy as np
n1=np.arange(10,1,-2)
print("\n A sequential array with a negative step :\n",n1)
n2=n1[np.array([3,1,2])]
print("\nElements at these indices are :\n",n2)
x1=np.array([1,2,3,4,5,6,7,8,9])
arrr=x1[np.array([1,3,-3])]
print("\n Elements are :\n",arrr)
a8=np.arange(20)
print("\n Array is :",a8)
print("\na8[-8:17:1]=",a8[-8:17:1])
print("\na8[10:]",a8[10:])
print("\nReverse order :",a8[::-1])
a9=np.array([[1,2,3],
             [4,5,6],
             [7,8,9],
             [10,11,12]])
print("\nRow indexing :",a9[...,0],a9[...,1],a9[...,2])
a10=np.array([[1,2],[3,4],[5,6]])
print(a10[[0,1,2],[0,0,1]])
a11=np.array([10,40,80,50,100])
print("\nGreater than 50 : ",a11[a11>50])
a12=np.array([10,40,50,100])
print("\na12=",a12[a12%40==0]**2) 
b10=np.array([[5,5],
             [4,5],
             [16,4]])
sumrow=b10.sum(-1)
print(b10[sumrow%2])


# In[ ]:


import numpy as np
ri=np.array([[1.9,2,3],
            [4,5,6],
            [7,8,9]])
print(ri)


# In[ ]:


print(ri[0,0])


# In[ ]:


print(ri[0,1])


# In[ ]:


print("The value of ri[0,0] is :", r[0,0],"and the value of ri[1,2] is :",ri[1,2])


# In[ ]:


print(np.sqrt(ri))


# In[ ]:


print(ri.T)


# In[ ]:


print(np.linalg.matrix_power(ri,3))


# In[ ]:


print(np.linalg.matrix_rank(ri))


# In[ ]:


print(np.linalg.inv(ri))


# In[ ]:


print(np.linalg.det(ri))


# In[ ]:


print(np.trace(ri))


# In[ ]:


import math as m
print(m.ceil(ri[0,0]))


# In[ ]:


print(m.floor(ri[0,0]))


# In[ ]:


print(m.pi)


# In[ ]:


print(m.pi*(ri))


# In[2]:


r2=np.arange(1,9)
r=np.zeros(9)
print(r)
print(r2)


# In[ ]:


r3=[1,2,3,4,5,6,7]
r4=[10,9,8,7,6,5,4]
s1=np.array(r3)
s2=np.array(r4)
print(s1*s2)


# In[1]:


import numpy as np
r5=np.arange(10,1)


# # Numpy | ndarray ......

# In[ ]:


import numpy as np
#creating a array object...
m=np.array([[1.0,2,3],
            [4,2,5]])
print("Array is a type of :",type(m))
print("Adress of m is :",id(m))
print("Array Dimension :",m.ndim)
print("Shape of array :",m.shape)
print("Size of array :",m.size)
print("Array stores elements of type:",m.dtype)


# In[ ]:


import numpy as np
m1=np.array([[1,2,4],
             [5,8,7]],dtype="complex")
print("Array created using passed list:\n",m1)
n1=np.array((1,2,3))
print("Array created using passed tuples:",n1)
o1=np.zeros((3,4))
print("Array intailized with all zeros :\n",o1)
o2=np.ones((3,4))
print("Array intailized with all ones :\n",o2)
p1=np.full((3,3),6,dtype="complex")
print("an array intialized with all 6s.and type is complex\n",p1)
p2=np.full((3,3),6)
print("an array intialized with all 6s:\n",p2)


# In[ ]:


import numpy as np
m2=np.array([[-1,2,0,4],
             [4,-0.5,6,0],
             [3,7,4,2.0]])
m3=m2[:2,::2]
print("Array with using first row and alternates"
                        "columns(0 and 2):\n",m3)
m4=m2[[0,1,2],[2,1,0]]
print(m4)
cond=m2>0
m4=m2[cond]
print("Elements greater than 0:\n",m4)


# In[ ]:


import numpy as np
m5=np.array([1,2,5,3])
print("Adding 1 elements in each :",m5+1)
print("Subtract -2 elements in each :",m5-2)
print("Multiply 3 in each elements :",m5*3)
print("Square each elements :",m5**2)
m5*=2
print("Double each element of original elements :",m5)
m6=np.array([[1,2,3],
             [3,5,6],
             [8,7,9]])
print("Original array is :\n",m6)
print("Transport of Array is :\n",m6.T)


# In[ ]:


import numpy as np
m7=np.array([[1,5,6],
            [4,7,2],
            [3,1,9]])
print("Largest number in m7{} array is :".format(m7),end="")
print(m7.max())
print("minimum elements is",m7.min())
print("sum of all arrays are:",m7.sum(axis=0))
print("Cumulative sum of along each row:\n",m7.cumsum(axis=0))


# In[ ]:


import numpy as np
m8=np.array([[1,2],
             [3,4]])
n2=np.array([[5,0],
             [7,1]])
print("Adding of array m8 :{} + array n2 : {}=".format(m8,n2),end="")
print(m8+n2)
print("Matrix multiplication :\n",m8.dot(n2))


# In[ ]:


import numpy as np
n3=np.array([0,np.pi/2,np.pi])
print(n3)
print("sine value of array element :",np.sin(n3))
n4=np.array([0,1,2,3])
print("Exponent of array elements is",np.exp(n4))
print("Square root of array is :",np.sqrt(n4))


# In[ ]:


import numpy as np
print(np.dtype(np.int16))
dt=np.dtype(">i4")
print("Byte order",dt.byteorder)
print("size is :",dt.itemsize)
print("Datatype name :",dt.name)


# # Numpy | Array Creation .....

# In[ ]:


arrrr=[1,2,3,4,5]
arrrr1=["geeks","for","geeks"]
for i in arrrr:
    print(i)


# In[ ]:


import array
l=array.array("i",[1,2,3])
print("The new created Array is :",end="")
for i in range(0,3):
    print(l[i],end="")
print("\r")


# In[ ]:


import numpy as np
n5=np.empty(2)
print("Matrix n5:\n",n5)
n6=np.empty([2,2])
print("Matrix n6 :\n",n6)
n7=np.empty([4,5],dtype=int)
print("matrix n7 :\n",n7)


# In[ ]:


import numpy as np
n8=np.arange(16)
print("Original array :\n",n8)
n8=np.arange(8).reshape(2,4)
print("Array with two rows and 3 columns",n8)
n8=np.arange(8).reshape(4,2)
print("Array with 4 rows and 9 columns",n8)
n8=np.arange(8).reshape(2,2,2)
print("Array with 3D shape",n8)


# In[ ]:


import numpy as np
print("\n",np.linspace(2.0,3.0,num=5,retstep=True),"\n")
n9=np.linspace(0,2,4)
print("\n",np.sin(n9))


# In[ ]:


import numpy as np
n10=np.array([[1,2],[3,4]])
n10.flatten()
print(n10)
n10.flatten("F")
print(n10)


# # Method For Array Creation In Numpy ......

# In[ ]:


#numpy.empty(shape,dtype=float,order="c") : return a new array of given shape and type with random values...
#numpy.empty_like(a,dtype=none,order="k",subok=True) : returns a new array and the same shape and type as given in the parameter...


# In[ ]:


import numpy as np
n11=np.empty(3,dtype=int)
print("Matrix of n11 is :\n",n11)
n12=np.empty([1,100],dtype=int)
print("Matrix of n12 array is :\n",n12)
n13=np.empty_like([2,5],dtype=int)
print("Matrix is :",n13)


# In[ ]:


#numpy.eye(R,c=none,k=0,dtype=type<"float>") : returns a matrix having 1's on the diagonal  and 0's elsewhere wrt K.....
import numpy as np
n14=np.eye(5,4,dtype=complex)
print("Matrix n14 is :\n",n14)
n15=np.eye(4,4,dtype=int)
print("\nMatrix is :\n",n15)


# In[ ]:


#np.identity(n,dtype=none) : Reurn  a identity matrix i.i., a square matrix with ones wit the main diagonal....
import numpy as np
n15=np.identity(4,dtype=float)
print("Matrix of n15 is :\n",n15)
n16=np.identity(100)
print("matrix of n16 is :\n",n16)


# In[ ]:


#np.ones() : returns a new array of given shapes and types with ones..
#np.ones_like : retuen the same shape and tyoe as given in the array...
n17=np.ones([2,3],dtype=int)
print("\n matrix of n17 is \n",n17)
n18=np.ones_like([2,5],dtype=int)
print("matrix of n18 is :",n18)
n19=np.arange(10).reshape(5,2)
n20=np.ones_like(n19,float)
print("Matrix of n20 is \n",n20)
print("size is :",np.size(n20))
print("sahpe is :",np.shape(n20))
print("item_size is :",n20.itemsize)
print("type is :",type(n20))
print("Adress is : ",id(n20))


# In[ ]:


#np.zeros : return a new array of given sahpe and type with zeros....
#np.zeros_like : return a new array with same size and type with zeros
n21=np.zeros([4,3],dtype=int)
print("matrix of n21 :\n",n21)
n23=np.arange(20).reshape(4,5)
n22=np.zeros_like(n23)
print("matrix of n23 is\n",n22)


# In[ ]:


#np.full() returms a new array with given shape and type as given in the array..
#np.full_like() returms a new array with given same shape and type as given in the array..
import numpy as np
n24=np.full([2,3],3,float)
print("Matrix of n24 is :\n",n24)
n25=np.arange(20,dtype=int).reshape(5,4)
print("Matrix of n26 is \n",np.full_like(n25,10))


# In[ ]:


#np.asarray() : function is used when we want to convert input to an array. input can be a lists,list of tuples, typles of ytuples,tuplesof lists and ndarrays..
import numpy as np
my_list=[1,2,3,7,9,0,8]
print("Input list :",my_list)
n26=np.asarray(my_list)
print("Output array from a input list is :",n26)
my_tuple=([1,3,9],[8,26])
print("Input tuple : ",my_tuple)
out_tuple=np.asarray(my_tuple)
print("Output array from a input tuple is :",out_tuple)


# In[ ]:


#asanyarray() function is used when we want to convert input to an array but it pass ndarray subclasses through.input can be scaler,lists,lists of tuples,tuples,tuples of tuples, tuples of lists and ndarrays...
import numpy as np
my_scalar=12
print("Input scaler is :",my_scalar)
out_arr1=np.asanyarray(my_scalar)
print("Output array from input scalar is :",out_arr1)


# In[ ]:


#np.ascontiguousarray() : function is used when we want to return a contiguous array in memory(C order)....
import numpy as np
my_input=100
print("Input Scalar :",my_input)
my_output=np.ascontiguousarray(my_input,dtype=float)
print("Output array from input scalar",my_output)
print(type(my_output))


# In[ ]:


#np.asmatrix() returns a matrix by interpreting the input as a matrix....
import numpy as np
h=np.matrix([[5,6,7],[4,6]])
print("via array like input \n",h,"\n")
i=np.asarray(h)
h[0,1]="x"
h[0,1]=2
print("i matrix :\n",i)


# In[ ]:


#copy() function return an array copy of the given object...
import numpy as np
p=np.array([[1,2,3],[2,3,4]])
print(np.copy(p))


# In[ ]:


#frombuffer() function interpret a buffer as 1-D arrays...
import numpy as np
p=np.array([[3,2],
            [4,5],
            [7,9]],dtype=float)
print("2- D array\n",p)
print("it will convert into 1-D array\n",np.frombuffer(p))


# In[ ]:


p1=np.fromfile("untitled.txt",dtype=complex)
print(p1)


# In[ ]:


p2=np.fromfunction(lambda i,j: i==j,(3,4),dtype=int)
print(p2)
p3=np.fromfunction(lambda i,j: i * j,(3,4),dtype=int)
print(p3)


# In[ ]:


p4=(x+x for x in range(5))
p5=np.fromiter(p4,float)
print("Array :",p5)


# In[ ]:


p6=np.fromstring("1,2,3,4,5",dtype=int,sep=",")
print(p6)


# In[ ]:


from io import StringIO
p7=StringIO("0 1 2 3\n4 5 6 7")
p8=np.loadtxt(p7)
print(p8)


# In[ ]:


p9=np.linspace(2,3,1000,dtype=float)
print(p9)


# In[ ]:


p10=np.logspace(2.0,3.0,5)
print(p10)


# In[ ]:


print(np.geomspace(2.0,9.0,1000))


# In[ ]:


import numpy as np
p11=np.matrix([[1,2,3],
               [4,5,6],
               [7,8,9]])
print("Main Diagonal Elements :\n",np.diag(p11))
print("Diagonal above main diagonal :\n",np.diag(p11,1))
print("Diagonal below main diagonal :\n",np.diag(p11,-1))


# In[ ]:


import numpy as np
print("Diagflat use on main diagonal :\n",np.diagflat([1,7,5,3]))
print("Diagflat above main diagonal :\n",np.diagflat([1,7,5],2))


# In[ ]:


import numpy as np
print("tri with k=1:\n",np.tri(3,3,-1,dtype=float))


# In[ ]:


import numpy as np
p12=np.matrix([[1,2,3],
               [2,4,5],
               [3,4,7]])
print("main daigonal elements :\n",np.tril(p12))
print("main daigonal elements :\n",np.tril(p12,1))
print("main daigonal elements :\n",np.tril(p12,-1))
print("main daigonal elements :\n",np.triu(p12))
print("main daigonal elements :\n",np.triu(p12,1))
print("main daigonal elements :\n",np.triu(p12,-1))


# In[ ]:


import numpy as np
p13=np.mat("4 1; 22 1")
p14=np.mat("3 2; 26 6")
p15=np.mat("7 5; 52 8")
print("mat p13\n",p13)
print("mat p14\n",p14)
print("mat p15\n",p15)
p16=np.bmat([[p13,p14],[p15,p13]])
print("bmat p16\n",p16)


# # Numpy | Data Type Objects ......

# In[ ]:


import numpy as np
print(np.dtype(np.int16))


# In[ ]:


import numpy as np
dt=np.dtype(">c8")
print("Byte order is :",dt.byteorder)
print("Size is :",dt.itemsize)
print("Data type is :",dt.name)


# In[ ]:


import numpy as np
p17=np.array([1])
print("Type is",type(p17))
print("Dtype is :",p17.dtype)
print("Id is :",id(p17))


# In[ ]:


import numpy as np
p18=np.dtype([("name",np.unicode_,16),
             ("grades",np.float64,(2,))])
p19=np.array([("sarah",(8.0,7.0)),
             ("john",(6.0,7.0))],dtype=p18)
print(p19[0])
print(p19[1])
print("Grades of john are:",p19[1]["grades"])
print("Grades of sarah are:",p19[0]["grades"])
print("Names are:",p19["name"])


# # Data Type Object(dtype) in Numpy ........

# In[ ]:


import numpy as np
li=[1,2,3]
p20=np.array([1,2,3])
print("Type is :",type(p20))
print("Type is :",type(li))
print("dtype is :",p20.dtype)


# In[ ]:


import numpy as np
p21=np.dtype([("name",np.unicode_,16),("grades",np.float,(2))])
print(p21["grades"])
print(p21["name"])


# # Numpy | Iterating over Array .....

# In[ ]:


#Python program for iterating over Array .....
import numpy as np
p22=np.arange(12)
p22=p22.reshape(3,4)
print("Original Array is :\n",p22)
print("Modify array is :\n")
for p23 in np.nditer(p22):
    print(p23,end=",")


# In[ ]:


#Python program for iterating over transpose array .....
import numpy as np
p24=np.arange(12).reshape(3,4)
print("Original Array",p24)
p25=p24.T
print("Modify array is :",end="")
for p26 in np.nditer(p25):
    print(p26,end="")


# In[ ]:


#python program for iterating over array using particular order....
import numpy as np
p27=np.arange(12).reshape(3,4)
print("Original Array",p27)
print("Modify Array in F-style order :",end="")
for p28 in np.nditer(p27,order="F"):
    print(p28,end="")


# In[ ]:


#python program for iterating over array using particular order....
import numpy as np
p27=np.arange(12).reshape(3,4)
print("Original Array",p27)
print("Modify Array in C-style order :",end="")
for p28 in np.nditer(p27,order="C"):
    print(p28,end="")


# In[ ]:


#Python program to modify the array values ...
import numpy as np
p29=np.arange(12).reshape(4,3)
print("Original Values :\n",p29)
for p30 in np.nditer(p29):
    p30=5*p29
print("Modifying Array is :\n",p30)


# In[ ]:


import numpy as np
p31=np.arange(12).reshape(4,3)
print("Original Values :\n",p31)
print("Modifying Vlaues :\n",p31*5)


# In[ ]:


#python program to iterating array values using external loop ...
import numpy as np
p32=np.arange(12).reshape(4,3)
print("Orriginal Array\n",p32)
print("Modifying array is:",end="")
for p33 in np.nditer(p32, flags=["external_loop"],order="C"):
print(p33,end="")


# In[ ]:


j=np.random.uniform(0,5,size=1000000)
print(j.mean())
print(np.zeros([3,3]),"\nshape is :",np.shape(np.zeros([3,3])))


# In[ ]:


u=np.array([[1,1.2,6],
            [5,8,7],
            [7,9,0]],dtype=bool)
print(u[-1])
print(u[...,0],u[...,1],u[...,2])
p=np.identity(5,dtype=int)
print(p)
print(id(p))
p=np.identity(5,dtype=bool)
print(p)
print("id is :",id(p),"type is :",type(p),"data type is :",p.dtype,"Dimension of Array :",p.ndim,"\n0 Row :",p[...,0],"0 column :",p[0,...])
p=np.eye(5,5,dtype=bool)
print(p)
print(id(p))


# In[ ]:


import numpy as np
p=np.random.random([10,20])
print(p,"\n")
print(np.random.randn(3))
print(np.random.randint(8))
print(np.random.uniform(10))


# In[ ]:


p=np.array([[1,2,3],[4,6,8],[4,6,8]])
print(np.triu(p,1))
print(np.tril(p,0))
print(np.tri(3,3,1,dtype=int))


# In[ ]:


import numpy as np
p=np.matrix([1,2,3,4])
print(p)
print("Min index :",p.argmin())#returns the min index
print("max index :",p.argmax())#returns the max index
print(p.mean())#mean
print(p.cumsum())#cumulative sum
print(p.cumprod())#cumlative product
print(p.var())#variance
print(p.std())#standard deviation
print(p.T)#Transport


# In[ ]:


# we can also use @ to take the inner product of two flats arrays...and we can also use the np.dot function ......
m=np.ones((2,2))
n=np.ones((2,2))
print(m@n)
print(np.dot(m,n))
print(m)
print(n)
a=np.array([1,2])
b=np.array([3,4])
print(a@b)
print(np.dot(a,b))


# # Numpy | Binary Operation ......

# In[6]:


#np.bitwise_and() function is used to compute the bit wise And of two array elememnt wise .....
import numpy as np
t=10
t1=11
print("Input number 1 :",t)
print("Input number 2 :",t1)
t3=np.bitwise_and(t,t1)
print("Bitwise_and of 10 and 11 is :",t3)
print("\n")
t4=[2,8,125]
t5=[3,3,115]
print("input array 1 :",t4)
print("Input array 2 :",t5)
print("Bitwise_and array 1 and array 2 is:",np.bitwise_and(t4,t5))
t6=np.array([[1,4,7],
             [4,9,2],
             [5,0,3]])

t7=np.array([[9,0,5],
             [3,5,8],
             [2,0,7]])
print("t6 array:\n",t6)
print("t7 array :\n",t7)
print("bitwise_and of t6 and t7 is:\n",np.bitwise_and(t6,t7))
            


# In[7]:


#np.bitwise_or() function is used to compute Bitwise or of twoarray element-wise.....
t8=10
t9=11
print("Input number 1 :",t8)
print("Input number 2 :",t9)
t3=np.bitwise_or(t8,t9)
print("Bitwise_or of 10 and 11 is :",t3)
print("\n")
t10=[2,8,125]
t11=[3,3,115]
print("input array 1 :",t10)
print("Input array 2 :",t11)
print("Bitwise_or array 10 or array 11 is:",np.bitwise_or(t10,t11))


# In[8]:


#np.bitwise_xor() : function is used to compute the bitwise XOR of two element- wise...
t12=10
t13=11
print("Input number 1 :",t12)
print("Input number 2 :",t13)
t14=np.bitwise_xor(t12,t13)
print("Bitwise_xor of 12 and 13 is :",t14)
print("\n")
t15=[2,8,125]
t16=[3,3,115]
print("input array 1 :",t15)
print("Input array 2 :",t16)
print("Bitwise_xor array 1 and array 2 is:",np.bitwise_xor(t15,t16))


# In[9]:


#np.invert() function is used to compute the bit wise inversion of an array element wise....
import numpy as np
t17=12
print("Input number",t17)
t18=100
print("Input number",t18)
t19=120
print("Input number",t19)
t20=127
print("Input number",t20)
print("Inversion of 12 is:",np.invert(t17))
print("Inversion of 100 is:",np.invert(t18))
print("Inversion of 120 is:",np.invert(t19))
print("Inversion of 127 is:",np.invert(t20))


# In[89]:


#np.left_shift() fuction is used to shift the bits of an integer to the left...
t21=6
t22=3
print("Input number",t21)
print("Number of bit shift",t22)
print("After left shifting of two bits is:",np.left_shift(t21,t22))


# In[ ]:


#np.left_shift() fuction is used to shift the bits of an integer to the Right...
t23=20
t24=2
print("Input number",t23)
print("Number of bit shift",t24)
print("After right shifting of two bits is:",np.right_shift(t23,t24))


# In[ ]:


#np.binary_repr(number,width=none) function is used to represent binary form of the input number as a string....
import numpy as np
t25=10
print("Input number is :",t25)
print("Binary reprenation of 10 is:",np.binary_repr(t25))
print("\n")
t26=[5,-8]
print("Input array:",t26)
print("Binary reprsenation of 5:",np.binary_repr(t26[0]))
print("Binary reprsenation of 5 using width parameter:",np.binary_repr(t26[0],width=50))
print("Binary reprsenation of -8:",np.binary_repr(t26[1]))
print("Binary reprsenation of -8 using width parameter :",np.binary_repr(t26[1],width=8))


# In[ ]:


#np.packbits(myarray,axis=none) function is used to pack the elements of binary valued into bits in a uint8 array....
import numpy as np
t27=np.array([[1,0,1],
              [0,1,0],
              [1,1,0],
              [0,0,1]])
print("Packing elements of array is\n",np.packbits(t27,axis=0))


# In[ ]:


#np.unpack_bits(myarray,axis=none) function is used to unpack elements of a uint8 into a binary valued output array.....
import numpy as np
t28=np.array([[5],[7],[23]],dtype=np.uint8)
print("Unpacking elements of array is\n",np.unpackbits(t28,axis=1))


# In[ ]:


import numpy as np
print(np.version)
print(np.Tester)
print(np.array([1,2,3],dtype="float32"))
print(np.array([range(x,x*3)for x in[2,4,6]]))
print(np.arange(0,200,2).reshape(10,10))
print(np.random.normal(0,1,(13)))
print(np.int16())
print(np.random.randint(10,size=6))
print(np.random.randint(10,size=(3,5)))
print(np.random.randint(10,size=(1,4,5)))


# In[ ]:


#Modifying values using any of the above index notation.....
import numpy as np
k=np.matrix([[1,9,7,6],
             [6,7,4,5],
             [7,4,7,9],
             [5,4,7,9]],dtype=float)
print(k)
k[0,0]=12
print(k)
k[0,1]=3.1478
print(k)
print(k[0,1])


# In[ ]:


import calendar
print(calendar.calendar(2019))


# In[ ]:


import numpy as np
u=np.arange(10)
print(u)
print(u[0:5])#first five elements...
print(u[5:])#After index 5...
print(u[4:7])#middle subaarays...
print(u[::2])#every other elements...
print(u[1::2])#every other element,starting at index 1...
print(u[::-1])#All elements Reversed...
print(u[5::-1])#Reversed every other from index 5....
print("\n")
o=np.arange(20).reshape(4,5)
print(o)
print(o[:2,:3])#two rows,three coloumns....
print(o[:3,::2])#three rows,every other column...
print(o[::-1,::-1])#finally,subarrat dimensions can even be reversed together...
#first row of o....
print("Rows")
print(o[:,0])
print(o[...,0])
print(o[...,1])
print(o[...,2])
print(o[...,3])
print(o[...,4])

#first coloumn of o...
print("Columns")
print(o[0,...])
print(o[0,:])
print(o[0])
print(o[1])
print(o[2])
print(o[3])


# # Numpy | Mathematical Function .......
#               
#               Trignometric Functions.... 

# In[ ]:


import numpy as np
import math
import matplotlib.pyplot as plt
print([0,math.pi/2.0,math.pi/3.0,np.pi])
#sin()function....
sin_values=np.arange(10).reshape(2,5)
print("Orginal array...\n",sin_values)
print("sin values in array:")
print(np.sin(sin_values))
#cos() function....
print("Cosins values of array is :")
print(np.cos(sin_values))
#tan() function.....compute tangent elements wise...
print("Tangent values of Array")
print(np.tan(sin_values))
#arcsin() function helps user to calculate inverse sine for all x...
arcsin_values=np.matrix([[1,0,-3],[4,-5,-7],[6,-0,-7]])
print("inverse sine values\n",np.arcsin(arcsin_values))
print("\nPython program showing graphical representation of arcsin() function.....\n")
in_array=np.linspace(-np.pi,np.pi,12)
out_array1=np.sin(in_array)
out_array2=np.arcsin(in_array)
print("in_array :",in_array)
print("\nut array with sin :",out_array1)
print("\nut array with arcsin:",out_array2)
print(math.isnan(out_array2[3]))
print(math.isnan(out_array2[5]))
plt.plot(in_array,out_array2,color="red",marker="o")
plt.plot(in_array,out_array1,color="blue",marker="*")
plt.title("blue:np.sin()\nred: np.arcsin()")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[ ]:


#arcos() function : help user to calculate inverse cos for all X.....
import numpy as np
import matplotlib.pyplot as plt
q=[0,1,0.3,-1]
print("Input arrays :",q)
print("Inverse cosin values",end="")
print(np.arccos(q))
#python program to showing graphical represenation of arccos function.....
in_array=np.linspace(-np.pi,np.pi,12)
out_array1=np.cos(in_array)
out_array2=np.arccos(in_array)
print("in_array :",in_array)
print("\nut array with cos :",out_array1)
print("\nut array with arccos:",out_array2)
print(math.isnan(out_array2[3]))
print(math.isnan(out_array2[5]))
plt.plot(in_array,out_array2,color="red",marker="o")
plt.plot(in_array,out_array1,color="blue",marker="*")
plt.title("blue:np.cos()\nred: np.arccos()")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[ ]:


#arctan() function helps to user to calculate inverse tangent for all x...
import numpy as np
import matplotlib.pyplot as plt
q=[0,1,0.3,-1]
print("Input arrays :",q)
print("Inverse arctan values",end="")
print(np.arctan(q))
#python program to showing graphical represenation of arctan function.....
in_array=np.linspace(-np.pi,np.pi,12)
out_array1=np.tan(in_array)
out_array2=np.arctan(in_array)
print("in_array :",in_array)
print("\nut array with tan :",out_array1)
print("\nut array with arctan:",out_array2)
print(math.isnan(out_array2[3]))
print(math.isnan(out_array2[5]))
plt.plot(in_array,out_array2,color="red",marker="o")
plt.plot(in_array,out_array1,color="blue",marker="*")
plt.title("blue:np.tan()\nred: np.arctan()")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
#arctan2()
ar1=[-1,+1,+1,-1]
ar2=[-1,-1,+1,+1]
ar3=np.arctan2(ar2,ar1)*180/np.pi
print("X-coordinates",ar1)
print("Y-coordinates",ar2)
print("arctan2 values:",ar3)
ar4=np.arctan2([0.,0.,np.inf],[+0.,-0.,np.inf])
ar5=np.arctan2([1.,-1.],[0.,0.])
print("arctan 4 :",ar4)
print("arctan 5 :",ar5)
print("Checking its inf or not:",np.isinf(ar4[2]))


# In[ ]:


import numpy as np
import math
import matplotlib.pyplot as plt
#np.degrees() function helps user to convert angle from radians to degree....
ar6=[0,math.pi/2,np.pi/3,np.pi]
print("Radian Values :\n",ar6)
print("Degree values is :",np.degrees(ar6))
print("Radiant values is:",np.radians(ar6))
#python program explaining rad2deg() function...or deg2reg()function....
print("Radians to Dregree...",np.rad2deg(ar6))
print("Degree to Radiant is :",np.deg2rad(ar6))
#hypot() function : helps user to calculate hypotenuse of right angled triangle....
ar7=[12,3,4,6]
ar8=[5,4,3,8]
print("ar7 :",ar7)
print("ar8 :",ar8)
ar9=np.hypot(ar7,ar8)
print("The unwrap value is :",end="")
print(np.unwrap(ar10))
print("Hypotenus is as follows :\n",ar9)
plt.plot(ar6,color="blue",marker="*")
plt.plot(ar4,color="cyan",marker="o")
plt.plot(ar5,color="red",marker=".")
plt.plot(np.deg2rad(ar6),color="yellow",marker="*")
plt.plot(ar9,color="Black",marker="o")
plt.title("Degree : Blue .. ar4 : Cyan .. ar5 : Red ...Radiant :Yellow ... hypot:Black")
plt.xlabel("Number of Degrees")
plt.ylabel("Numbers of Degrees") 
plt.show()


# # Hyperbolic Functions .......

# In[ ]:


import numpy as np
import math
#Hyperbolic function :
#sinh() function helps user to calculate hyperbolic sine for all x....
hy=np.linspace(-np.pi,np.pi,12)
print("Orginal value of hy is :",hy)
print("sinh of hy is :",np.sinh(hy))
#arcsinh() function help user to calculate inverse hyperbolic sine elements for all.....
print("arcsinh of hy is :",np.arcsinh(hy))
plt.plot(np.sinh(hy),color="cyan",marker="*")
plt.plot(np.arcsinh(hy),colodata:image/png;base64,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r="Black",marker="o")
plt.title("> arcsinh :Black \n > sinh :Cyan")
plt.xlabel("...............................X...........................................")
plt.ylabel("...............................Y...........................................")
plt.show()


# In[ ]:


import numpy as np
import math
#Hyperbolic function :
#cosh() function helps user to calculate hyperbolic cosin for all x....
hy=np.linspace(-np.pi,np.pi,12)
print("Orginal value of hy is :",hy)
print("sinh of hy is :",np.cosh(hy))
#arccosh() function help user to calculate inverse hyperbolic arccosh elements for all.....
print("arccosh of hy is:",np.arccosh(hy))
plt.plot(np.cosh(hy),color="Black",marker="*")
plt.plot(np.arccosh(hy),color="Cyan",marker="o")
plt.title("> cosh :Black \n >")
plt.xlabel("...............................X...........................................")
plt.ylabel("...............................Y...........................................")
plt.show()


# In[ ]:


import numpy as np
import math
#Hyperbolic function :
#tanh() function helps user to calculate hyperbolic tanh for all x....
hy=np.linspace(-np.pi,np.pi,12)
print("Orginal value of hy is :",hy)
print("tanh of hy is :",np.tanh(hy))
#arctanh() function help user to calculate inverse hyperbolic arctanh elements for all.....
print("arctanh of hy is:",np.arctanh(hy))
plt.plot(np.tanh(hy),color="Red",marker="*")
plt.plot(np.arctanh(hy),color="Blue",marker="o")
plt.title("> tannh :Red \n > arctanh :Blue")
plt.xlabel("...............................X...........................................")
plt.ylabel("...............................Y...........................................")
plt.show()


# # Rounding Functions .......

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
#around()/round() function helps user to to evenly round array elements to the given number of decimals...
rd=[.5,1.5,2.6,3.5,4.5,10.1]
print("rd array",rd)
print("Rounding of rd",np.around(rd))
rd1=[.535,1.6785,2.785,3.675,4.675,10.111]
print("rd1 array",rd1)
print("Rounding of rd1",np.around(rd1))
rd2=[2.53,3.345,4.455,33.5,4.55,8.671]
print("rd2 array",rd2)
print("Rounding of rd2",np.around(rd2))
rd3=[4.65,7.095,2.65,6.65,7.5,3.1]
print("r3 array",rd3)
print("Rounding of rd3",np.around(rd3))
print("Rounding of rd3",np.round(rd3))
plt.plot(np.around(rd),color="black",marker="*")
plt.plot(np.around(rd1),color="blue",marker="*")
plt.plot(np.around(rd2),color="cyan",marker="*")
plt.plot(np.around(rd3),color="yellow",marker="*")
plt.title("\t rd:black  \t rd1:blue\n  \t rd2:cyan  \t rd3:yellow")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
print(end="")


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
#rint() function round elements of the array to the nearest integer.....
rd4=[10.3,2.5,13.5,6.5,10,4]
print("rd4 array is :",rd4)
print("Rounded value is :",np.rint(rd4))
plt.plot(np.rint(rd4),color="black",marker="*")
plt.title("\t rd4 :Black")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
#fix() function round elements of the array to the nearest integer towards 0...
rd5=[.5,1.5,2.6,3.5,4.5,10.1]
print("rd array",rd5)
print("Rounding of rd5",np.fix(rd5))
rd6=[.535,1.6785,2.785,3.675,4.675,10.111]
print("rd6 array",rd6)
print("Rounding of rd6",np.fix(rd6))
rd7=[2.53,3.345,4.455,33.5,4.55,8.671]
print("rd7 array",rd7)
print("Rounding of rd7",np.fix(rd7))
rd8=[4.65,7.095,2.65,6.65,7.5,3.1]
print("r8 array",rd8)
print("Rounding of rd8",np.fix(rd8))
plt.plot(np.around(rd5),color="black",marker="*")
plt.plot(np.around(rd6),color="blue",marker="*")
plt.plot(np.around(rd7),color="cyan",marker="*")
plt.plot(np.around(rd8),color="yellow",marker="*")
plt.title("\t rd5:black  \t rd6:blue\n  \t rd7cyan  \t rd8yellow")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


# In[ ]:


import numpy as np 
import matplotlib.pyplot as plt
#floor() function : return the floor of the input,element wise...
rd9=[4.3,10.6,2.2,3.8,5.9,12.7,4.4]
print("Rd9 array is :",rd9)
print("Floor value of rd9 is :",np.floor(rd9))
#ceil() function : return the ceil of the input,element wise...
rd10=[10.3,7.6,8.2,9.8,8.9,2.7,7.4]
print("Rd10 array is :",rd10)
print("Floor value of rd10 is :",np.ceil(rd10))
#trunc() function : return the truncate value of the elements of array...
rd11=[8.9,7.8,6.8,0.0,4.5,1.2]
print("Rd11 array is :",r11)
print("truncate of rd11 is",np.trunc(rd11))
plt.plot(np.floor(rd9),color="Red",marker="o")
plt.plot(np.floor(rd10),color="Blue",marker="*")
plt.plot(np.floor(rd11),color="cyan",marker="*")
plt.title("\t rd9:Red \t rd10:Blue \t rd11:cyan")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


# # Exponents and Logarithms Functions ........

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
#exp() function : helps user to calculate exponential of all the elements in the input array....
print("The value of e is:",np.e)
w=[1,3,5]
print(w)
print("output is :",np.exp(w))
#log() function : helps user to calculate natural logarithm of x where x is belongs to the input of array elements.....
w1=[1,3,5,2**8]
print("input of w1 is :",w1)
print("output of w1 is :",np.log(w1))
#expm1() function helps user to calculate expoential of all the elements subtracting 1 from all the input array elements...
w2=[1,3,5]
w3=np.exp(w2)
print("exponential value of array elemts is",w3)
print("exponential value of array elements -1",np.expm1(w3))
#exp2() function helps user to calculate 2**x for all x being the array elements...
w4=np.exp2(w3)
print("2**x values :",w4)
#log10() function helps user to calculate base 10 log of x....
w5=[1,3,4,6,10**8]
print("w5 is",w5)
w6=np.log10(w5)
print("Logarthim value of w6 is :",w6)
#log2() function helps user to calculate bse 2 log of x...
w7=[1,3,5,2**8]
print("w7 is :",w7)
w8=np.log2(w7)
print("log2 value of w8 is:",w8)
#log1p() function helps user to calculate natural logarthic value of x+1....
w9=[2,4,7,8]
print("w9 is",w9)
w10=np.log1p(w9)
print("log of w10 +1",w10)
#logaddexp() function is used to calculate logarithm of the sum  of exponents of the inputs....
w11=2
w12=3
print("w11 is",w11)
print("w12 is",w12)
w13=np.logaddexp(w11,w12)
print("output is",w13)
#logaddexp() function is used to calculate logarithm of the sum  of exponentrs of the inputs in base 2....
w14=2
w15=3
print("w14 is",w14)
print("w15 is",w15)
w16=np.logaddexp2(w14,w15)
print("output is",w16)

plt.plot(w16,color="blue",marker="*")
plt.plot(w13,color="green",marker="*")
plt.plot(w10,color="red",marker="*")
plt.plot(w8,color="cyan",marker="*")
plt.plot(w6,color="yellow",marker="*")
plt.plot(w3,color="pink",marker="*")
plt.plot(np.log(w1),color="black",marker="*")
plt.plot(np.exp(w),color="purple",marker="*")
plt.title("\t w3 : pink \tw6 : yellow \t w8 : cyan \t w : purple \n \t w10 : red \t w13 : green \t w16 : blue \t w1 : black ")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


# # Arthmetic functions ......

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
#reciprocal() function is used to calculate reciprocal of all the elements in the input array....
t=2.0
print("Input:",t)
t1=np.reciprocal(t)
print("Reciprocal output is",t1)
#divide()function is used to divide first array by elements from second elements....
t2=10
t3=2
t4=np.divide(t2,t3)
print("Divide is",t4)
#add() function is used when we want to addition of two arrays.......
t5=5
t6=9
t7=np.add(t5,t6)
print("Addition of t7 is :",t7)
#negative() function is used when we want to compute the negative of array elements....
t8=10
t9=np.negative(t8)
print("negative aray elements are:",t9)
#positive() function is used when we want to compute the positive of array elements....
t10=[-10]
t11=np.positive(t10)
print("positive array elements are:",t11)
#multiply() function is used to multiply first array by elements from second elements....
t12=89.88
t13=78
t14=np.multiply(t12,t13)
print("Multiply array elemets is:",t14)
#power() function is used to raised to power of first elements to seconf elements....
t15=3
t16=3
t17=np.power(t15,t16)
print("raised to the power:",t17)
#subtract() function is used to subtract first array by elements from second elements....
t18=89.88
t19=78
t20=np.subtract(t18,t19)
print("subtract array elemets is:",t20)
#true_divide() function is used to return true division of the input elements wise.......
t21=100
t22=4
t23=np.true_divide(t21,t22)
print("true_divide array elemets is:",t23)
#floor_divide() function is used to return the largest integer smaller or return  equal to the divsion of the inputs.....
t24=[2,7,3,11,4]
t26=2
t25=np.floor_divide(t24,t26)
print("floor_divide array elemets is:",t25)
#float_power() function is used to raised to power of first elements to second elements....
t27=3
t28=3
t29=np.float_power(t27,t28)
print("raised to the power:",t29)
#mod() function is used to remaindar of division....
t30=198
t31=10
t32=np.mod(t30,t31)
print("remainder is",t32)
#remainder() function is used to remaindar of division....
t33=198
t34=10
t35=np.remainder(t33,t34)
print("remainder is",t35)
#divmod() function return elements wise quotient and remainder simultanseously.....
t36=198
t37=10
t38=np.divmod(t36,t37)
print("Divide & remainder is",t38)


plt.plot(t1,color="brown",marker="o")
plt.plot(t4,color="green",marker="*")
plt.plot(t7,color="yellow",marker="o")
plt.plot(t9,color="blue",marker="*")
plt.plot(t11,color="black",marker="o")
plt.plot(t14,color="pink",marker="*")
plt.plot(t17,color="grey",marker="o")
plt.plot(t20,color="silver",marker="*")
plt.plot(t23,color="gold",marker="*")
plt.plot(t25,color="purple",marker="*")
plt.plot(t29,color="cyan",marker="*")
plt.plot(t32,color="orange",marker="*")
plt.plot(t35,color="darkgreen",marker="*")
plt.plot(t38,color="lightblue",marker="*")
plt.title("\t t1: brown \t t4 : Red  \t t9 : ")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


# # Complex Number Functions ......

# In[ ]:


import numpy as np
#isreal() function test elements wise whether it is real number or not and return the result as a boolean array....
print("Is real :",np.isreal([1+1j, 0j ]))2
print("Is real :",np.isreal([1,0]))
#conj() function helps user to conjugate any complex number....
in_complx=2+4j
out_complx=np.conj(in_complx)
print("output conjugated complex number of 2",out_complx)
in_complx1=2-4j
out_complx1=np.conj(in_complx1)
print("output conjugated complex number of 2",out_complx1),


# # Special Functions .........

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
#cbrt() function helps user to calculate cube root x......
h=[1,27000,64,-1000]
h1=np.cbrt(h)
print("cube root of h is",h1)
#clip() function is used to limit the values in an array....
h2=[1,2,3,4,5,6,7,8]
h3=np.clip(h2,a_min=3,a_max=7)
print("output is",h3)
#convolve() function returns the discrete,linear convoultion of two one dimesion array squences....
h4=[2,4,6,8]
h5=[1,3,5,7]
h6=np.convolve(h4,h5)
print("output is",h6)
#sqrt() function is used to return non-negative square root of an array element wise....
h7=[5,6,9,3,6]
h8=np.sqrt(h7)
print("square root of h8 is:",h8)
#square() function is used to return the element-wise square of the input....
h9=[3,4,2,1,5]
h10=np.square(h9)
print("square of h9 is",h10)
#absolute() function helps user to calculate absolute value of each elements...
h11=[1,-3,15,-466]
h12=np.absolute(h11)
print("The Absolute value of h12 is:",h12)
#fabs() function compute the absloute value elements wise...
h13=[-23,60,-15,-13]
h14=np.fabs(h13)
print("The absolute value is:",h14)
#sign() function returns an elements-wise indication of the sign of a number...
h15=[-2,5,7,-7,-6]
h16=np.sign(h15)
print("output is",h16)
plt.plot(h3,color="black",marker="*")
plt.plot(h6,color="green",marker="*")
plt.plot(h8,color="blue",marker="*")
plt.plot(h10,color="yellow",marker="*")
plt.plot(h12,color="red",marker="*")
plt.plot(h14,color="cyan",marker="*")
plt.plot(h16,color="pink",marker="*")
plt.title("\t h3 : black \t h6 : green \t h8 : blue \t h10 : yellow \t h12 : red \t h14 : cyan \t h16 : pink")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


# # Numpy | String Operation .......

# In[ ]:


import numpy as np
import math
#lower() function returns the lowercase string from the given string...
st="AASHISH KUMAR"
print(st.lower())
print(st.islower())
print(np.char.lower(st))
print(np.char.islower(st))
#upper() function returns the uppercase string from the given string...
st1="aashish kumar"
print(st.upper())
print(st.isupper())
print(np.char.upper(st1))
print(np.char.islower(st1))
#split(): function returns a list of strings after breaking the given string by the specified seprator....
print(np.char.split("Aashish Kumar"))
print(np.char.split("aashish,kumar",sep=","))
#join( function is a string method and returns a string in which the elements of sequence have been joined by str seprator...
print(np.char.join("-","geeks"))
print(np.char.join(["-",":",","],["for","geeks","hi"]))
#strip() function isused to remove all the leading and trailing spaces from string...
print(np.char.strip("a  a sh is h"))


# In[52]:


#Different Ways to Reversed a String.....
a="hello"
print("-".join(reversed(a))) # use a rversed function with join method....... 

print(a[::-1]) # use slicing indexing.....

for i in reversed(a): # use for loop with reversed function...... 
    print(i,end="")


# In[51]:


import numpy as np
a=np.array([[1,2,3],[4,5,6],[6,7,8]])
print(a)
b=np.copy(a[0,2])
print(b)
c=np.copy(a[1,2])
print(c)


# # Numpy | Linear Algebra .........

# In[92]:


#MATRIX EIGENVALUES FUNCTIONS .........
import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt
#linalg.eigh() function is used to return the eigenvalues an eigenvector of a complex of a matrix.....
ei=np.array([[1,-2],[2j,5]])
print("Array ei is :",ei)
ei1,ei2=lin.eigh(ei)
print("eigen value is",ei1)
print("eigen value is",ei2)
#linalg.eig() function is used to compute the eigenvalues and right eigenvectors of a square array.....
ei3=np.diag([1,2,3])
print("Array is:\n",ei3)
ei4,ei5=lin.eig(ei3)
print("eigen value is:",ei4)
print("eigen value is:\n",ei5)
#linalg.eigvals() function is used to compute te eigenvalues of a general matrix.....
ei6=np.diag([-1,1])
print(ei6)
ei7,ei8=lin.linalg.eigvals(ei6)
print("eigen value is:",ei7)
print("eigen values is:",ei8)
#linalg.eigvalsh() function is used to compute te eigenvalues of a complex hermilitian or real symmetric ,matrix.....
ei9=np.diag([1,3,4])
print("Array is:\n",ei9)
ei10=lin.eigvalsh(ei9)
print("eigen values is:",ei10)
plt.plot(ei1,color="blue",marker="*")
plt.plot(ei2,color="green",marker="*")
plt.plot(ei4,color="cyan",marker="*")
plt.plot(ei5,color="black",marker="*")
plt.plot(ei7,color="pink",marker="*")
plt.plot(ei8,color="purple",marker="*")
plt.plot(ei10,color="yellow",marker="*")
plt.title("\t ei1 : blue \t ei2 : green \t ei4 : cyan \t ei5 : black \t ei7 : pink \t ei8 : purple \t ei10 : yellow")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


# In[3]:


#MATRIX & VECTOR PRODUCTS.......
import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt
#np.dot() function returns te dot product of vector a & b.....
product=np.dot(5,4)
print("Dot product of scaler values is:",product)
vector_a=2+3j
vector_b=4+5j
product1=np.dot(vector_a,vector_b)
print("Dot product of array is:",product1)
#np.vdot() function returns te dot product of vector a & b.....if first arguments is complex conjugate of the first arguments is used for calculations of the dot products.....
vector_a1=2+3j
vector_b1=4+5j
product2=np.vdot(vector_a1,vector_b1)
print("Dot product of array is:",product2)
#np.matmul() function returns matrix product of two arrays....
m=np.array([[1,0],
            [0,1]])
m1=np.array([1,2])
m2=np.matmul(m,m1)
print("matrix multiplication is:",m2)
m3=np.matmul(m1,m)
print("matrix multiplication is:",m3)
m4=np.matmul([2j,3j],[2j,3j])
print("matmul is:",m4)
m5=np.matmul([1,2j],[3,2])
print("matmul is",m5)
#np.inner() function returns the inner product of two arrays....
in1=np.arange(24).reshape(2,3,4)
in2=np.arange(4)
print(in1,in2)
in3=np.inner(in1,in2)
print("inner matrix of two array is :\n",in3)
in4=np.inner(np.eye(2),7)
print(in4)
in5=np.diag([7,7])
print(in5,type(in5),id(in5),np.shape(in5),np.size(in5))
#np.outer() function returns the outer product of two vectors......
out1=np.outer(np.ones((5,)),np.linspace(-2,2,5))
print("outer product of a vector is :\n",out1)
out2=np.outer(1j*np.linspace(2,-2,5),np.ones((5,)))
print("outer vector is:\n",out2)
out3=np.array(["a","b","c"],dtype=object)
out4=np.outer(out3,[1,2,3])
print("outer vector of letters is:\n",out4)
#linalg.multi_dot() function compute the dot product of two or more array in a single function call,while automatically selecting te fastest evaluation order...
e1=np.array([[1,3,8],
             [4,6,9]])
e2=lin.multi_dot(e1)
print("dot product of two dim array is..",e2)
#np.tensordot() function compute tensor dot product along specified axes for arrays >=1-D
e3=np.arange(60.).reshape(3,4,5)
e4=np.arange(24.).reshape(4,3,2)
print("array e3 is:\n",e3)
print("array e4 is:\n",e4)
e5=np.tensordot(e3,e4,axes=([1,0],[0,1]))
print("Tensordot product of a array is:\n",e5)
#np.einsum() function evalutes the Einstein summation convention on the operands....
e6=np.arange(25).reshape(5,5)
e7=np.arange(5)
e8=np.arange(6).reshape(2,3)
print(e6)
e9=np.einsum("ii",e6)
print("einsum of a6 array is",e9)
plt.plot(product,color="Red",marker="*")
plt.plot(product1,color="yellow",marker="*")
plt.plot(product2,color="blue",marker="*")
plt.plot(m2,color="green",marker="*")
plt.plot(m3,color="yellow",marker="*")
plt.plot(m4,color="pink",marker="*")
plt.plot(m5,color="black",marker="*")
plt.plot(in3,color="lightgreen",marker="*")
plt.plot(in4,color="cyan",marker="*")
plt.plot(product,color="lightblue",marker="*")
plt.plot(product,color="Red",marker="*")
plt.xlabel("X-axis")
plt.xlabel("Y-axis")
plt.show()




# In[13]:


#python program to print nXn checkboard pattern using numpy.....
import numpy as np
def printcheckboard(n):
    x=np.zeros((n,n),dtype=int)
    x[1::4, ::2] = 1
    x[::2, 1::4] = 1
    for i in range(n):
        for j in range(n):
            print(x[i][j],end="")
        print()
printcheckboard(8)


# # Numpy | Sorting, Searching, Counting .......

# In[21]:


import numpy as np
#np.sort() function returns a sorted copy of an array .....
a=np.array([[12,15],[10,1]])
a1=np.sort(a,0)
a2=np.sort(a,-1)
a3=np.sort(a,axis=None)
print("Along first axis..\n",a1)
print("Along last axis..\n",a2)
print("Along none axis..\n",a3)


# In[11]:


import pandas as pd
a = pd.Series([1,2,3,4,5,6])
b = pd.DataFrame([1,2,3,4,5,5])
print(b)


# In[ ]:




