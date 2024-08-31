#!/usr/bin/env python
# coding: utf-8

# ### 1.Numpy intro

# ###### Numpy : Numerical Python
# OPERATIONS using NUMPY
# .Mathematical & logical operations on arrays
# .Fourier transforms and routines for shape manipulation
# .Operations-algebra(in-built algebra functions. random number generation)

# ###### Numpyrepacementfor MATLAB

# In[1]:


pip install numpy


# ###### To install numpy,run the below command
# python setup.py install

# In[2]:


import numpy as np


# ###### Numpy-NDARRAY object
# 
# ndarray-N-dimensional array (items in this collection are same type and accessed using a zero  based index)
# basic ndarrayis created using an array function in numpy
# 
# 
# numpy.array

# In[3]:


np.array


# In[4]:


np.array(object, dtype=None, copy=True, order=None, subok=False, ndmin=0 )


# ###### 
# object :any object exposing the array interface method returns an array, or any(nested) sequence
# 
# 
# dtype :Desired data type of array,optional
# 
# copy : By defaut(True), the object is copied
# 
# order : C (row major), F(Column major), A(any default)
# 
# subok : By default, returned array forced to be a base cass array if true, sub-casses passed through 
# 
# 
# ndmin : specifies minimum dimensions of resutant array

# In[5]:


a=np.array([1,2,3])


# In[6]:


a


# In[7]:


print (a)


# In[8]:


#more than one dimension
b = np.array([[1,2,3],[4,5,6]])
b


# In[9]:


#minimum dimensions
import numpy as np
c= np.array([1,2,3,4,5],ndmin=2)
c


# In[10]:


c= np.array([1,2,3,4,5])
c


# In[11]:


#dtype parameter
d=np.array([1,2,3], dtype=complex)
d


# the ndarray object consists of contiguous one_dimensional segment of computer memory, combined with an indexing scheme that maps search item to a location in the memory block. the memory block holds the elements in a row major order (C style) or a column- major order (Forton or Matlab style)

# Numpy DATA types
# bool_
# int_
# intc
# intp(used for indexing
# int8(Byte -128 to 127
# int16(-32768to 32767
# int32(
# int64
# unit8(unsigned intiger
# unit16,unit32,unit64
# foat_,foat16,foat32,foat64
# complex_(imaginary
# complex64, complex128
# numpy numerical types are instances of dtype(data-type)objectseach having unique characteristics.the dtypes are avaiable as np.bool_,np.float32 etc

# In[12]:


np.bool_


# A datatype object describes interpretation of fixed block of memory corresponding to an array depending on the foowingaspects:
# Type of data,size of data, byte order, 
# 
# incase of structured type, the names of fields,data types of each fields and partof the memory block taken by each field
# 
# If fata type is a subarray, its shape and datatype
# 
# thebyte order is decided by prefixing < or >

# np.dtype(object, align, copy)
# 
# Object:  to be converted to data type object
# 
# Align:  If true, addspadding to thefied to make it simiar to c_struct
# 
# Copy: Makes a new copy of dtype object.If false, the result is reference to built_in data type object
# 

# In[13]:


#using array scalar type
import numpy as np
dt= np.dtype(np.int32)
dt


# In[14]:


#int8,int16, int32, int64,int128 can be replaced by equivaent string 'i1', 'i2', 'i4','i8'.....etc
import numpy as np 
dt = np.dtype('i1')
dt


# In[15]:


#using endiannotation
dt = np.dtype('<i4')
dt


# In[16]:


#using endiannotation
dt = np.dtype('>i4')
dt


# In[17]:


# structured data type fied nae and corresponding datatype
dt=np.dtype([('age',np.int16)])
dt


# In[18]:


dt= np.dtype([('age',np.int8)])
print(dt)


# In[19]:


#applyit to nd array object
dt= np.dtype([('age',np.int8)])
a = np.array([(10),(11),(20),(30)], dtype=dt)
print(a)


# In[20]:


# fienamecanbeused to access content of age coumn
print(a['age'])


# Define a structured data type called  STUDENT with a field'name', an INTEGER FILED 'age'and a FLOAT FIELD 'marks' this dtype is appliedto nd array object
# 

# In[21]:


import numpy as np
student = np.dtype([('name', 'S20'),('age','i1'),('marks','f4')])
print(student)


# In[22]:


import numpy as np
student=np.dtype([('name','S20'),('age','i1'),('marks','f4')])
a=np.array([('abc',21,50),('xyz',18,75)],dtype =student)


# In[23]:


a


# In[24]:


print(a)


# eachbuit indata type has a character code that uniquey identifis it 
# 'b' booean
# 'i' signedinteger
# 'u' unsigned integer
# 'f'floatingpoint
# 'c' complex floating point
# 'm' time delta
# 'M' date time
# 'O' pythonobject
# 'S', 'a'(byte-)string
# 'U' unicode
# 'V' raw data(void)
# 

# ### numpy-array attributes
# shape, size, data type, dimension
# 

# #### ndarray.shape
# 
# This array attribute returns a tuple consisting of array dimensions.aso be used to resize the array
# 

# In[25]:


import numpy as np
a=np.array([[1,2,3],[4,5,6]])
a.shape


# In[26]:


import numpy as np
b=np.array([[1,2,3],[2,4,5]])
print(b)
print("Theshapeis", b.shape)

b.shape=(3,2)
print(b)
print("newshape",b.shape)


# Numpy alsoprovides a reshape function to resize anarray
# 
# 

# In[27]:


b.reshape(2,3)


# ndarray.ndim
# 

# In[28]:


#an array of evenly spaced numbers
import numpy as np
c=np.arange(20)
c


# In[29]:


c.ndim


# It is a one dimensinal array
# 

# In[30]:


b.ndim


# In[31]:


d=c.reshape(2,5,2)


# In[32]:


d


# In[33]:


d.ndim


# ######  numpy.itemsize

# In[34]:


#dtype of arrayis int8(1byte)
x=np.array([1,2,3,4,5],dtype=np.int8)
x.itemsize


# In[35]:


d.itemsize


# In[36]:


#dtype of array is now float32(4bytes)
x=np.array([1,2,3,4,5],dtype=np.float32)
x.itemsize


# ###### numpy.fags
# 
# thendarrayobject hasthefoowingattributes.
# 
#   C_CONTIGUOUS (C): data is in a singe c style contiguous segment
# 
#   F_CONTIGUOUS(F) : data is in a singe fortran style contiguous segment
#   
#   OWNDATA (O): the array owns the memory it uses or borrows it from another object
#   
#   WRITEABLE (W): the da1ta area can be written to . setting this to false locs the data , making it read only
#   
#   ALIGNED(A) :Thw data and all elements are aligned appropriatley for the hardware
#   
#   WRITEBACKIFCOPY :
#   
#   UPDATEIFCOPY(U): this array is a copy of some other array when this arr1ay is deallocated, thebase array will be updated with the contents ofthis array

# In[37]:


print (x.flags)


# # Numpy array creation routines

# numpy.empty

# In[38]:


#it creates an uninitialized array of specified shape and dtype
#numpy.empty(shape, dtype=float, order= 'c')
x=np.empty([3,2], dtype= int)
x


# In[39]:


#numpy.zeros(shape, dtype=float, order= 'c')
x=np.zeros([3,2], dtype= int)
x


# In[40]:


x=np.zeros(5)
x


# In[41]:


#custom type
x= np.zeros((2,2), dtype = [('x','i4'),('y', 'i4')])
print(x)


# #### Numpy.ones
# 

# In[42]:


x=np.ones([3,2], dtype=int)
x


# In[43]:


#numpy.ones(shape, dtype=None, order='C')
x=np.ones([5])


# In[44]:


x


# In[45]:


x=np.ones([5], dtype = int)
x


# ##### Numpy  array from existing data

# How to create an array from existing data ,It is similary to numpy.array , it is useful for converting ython sequence into ndarray 
# 
# 
# numpy.asarray(a, dtype= None, order=none)

# In[46]:


#convert list into ndarray
x=[1,2,3]
a=np.asarray(x,dtype=float)


# In[47]:


print(a)


# In[48]:


x=[(1,2,3),(4,5)]
x


# In[49]:


a=np.asarray(x)
print(a)


# #### num.frombuffer(buffer, dtype= float, count=-1,offset = 0)
# buffer: an object that exposes buffer interface
# 
# dtype: data of returned ndarray. float default
# 
# count:the number of items to read default -1 means all data
# 
# offset: the starting position to read from default 0

# In[50]:


import numpy as np
s= 'Hello world'
a=np.frombuffer(s, dtype='S1')
a


# In[51]:


import numpy as np

s = 'Hello world'
s_bytes = s.encode('utf-8')  # Convert string to bytes
a = np.frombuffer(s_bytes, dtype='S1')
print(a)


# ### numpy.fromiter
# builds an ndarray object from any iterable object. a new one dimensional array is returned by this function
# 
# numy.fromiter(iterable, dtype, count=-1
# 

# In[52]:


# create list object using range function
import numpy as np
list1= range(5)
print (list1)


# In[53]:


list1


# In[54]:


import numpy as np

numbers = range(5)
numbers_list = list(numbers)  # Convert range to a list
print(numbers_list)


# In[56]:


# obtain iterator object from list
import numpy as np
list2 = range(5)
it=iter(list2)

#use iterator to create ndarray
x=np.fromiter(it,dtype=float)
print(x)


# #### Numpy array from numerical ranges
# 
# returns ndarray object containing evenly spaced values within a given range
# numpy.arange(start, stop, atep, dtype)

# In[58]:


import numpy as np
x=np.arange(5)
x


# In[60]:


#dtye set
x=np.arange(5, dtype=float)
print(x)


# In[62]:


#start stop set
import numpy as np
x = np.arange(10,20,2)
print(x)


# #### numpy.linspace
# similar to arange()function, The number of evenly spaced values between the interval is specified
# 
# numpy.linspace(start, stop, num, endpoint, retstep, dtype)

# The constructor takes the following parameters.
# 
# start:  The starting value of the sequence
# 
# stop : The end value of the sequence, included in the sequence if endpoint set to true
# 
# num :The number of evenly spaced samples to be generated. Default is 50
# 
# endpoint :True by default, hence the stop value is included in the sequence. If false, it is not included
# 
# retstep :If true, returns samples and step between the consecutive numbers
# 
# dtype :Data type of output ndarray

# In[65]:


x = np.linspace(10,20, 5)
print(x)


# In[ ]:





# In[ ]:





# In[ ]:




