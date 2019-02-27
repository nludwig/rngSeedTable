#code adapted by NBL from
#code adapted by Snack-X
#from code & theory due to
#M.E. O'Neill: see
#www.pcg-random.org
import numpy as np
cimport cython

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef unsigned int _uint32(unsigned int n):
  return n & 0xffffffff

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef unsigned long _uint64(unsigned long n):
  return n & 0xffffffffffffffff

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef unsigned long[:] _rng(unsigned long state=0, unsigned long inc=0):
  rngS=np.zeros(2,dtype=np.uint64)
  cdef unsigned long [:] rngSV=rngS
  #rngSV[0]=_uint64(state)
  #rngSV[1]=_uint64(inc)
  rngSV[0]=state
  rngSV[1]=inc
  return rngSV

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def rng(unsigned long state=0, unsigned long inc=0):
  return _rng(state,inc)

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _srandom_r(unsigned long [:] rngSV, unsigned long initstate, unsigned long initseq):
  initstate=_uint64(initstate)
  initseq=_uint64(initseq)

  rngSV[0]=0
  rngSV[1]=_uint64(initseq << 1) | 1
  _random_r(rngSV)
  rngSV[0]=rngSV[0]+initstate
  #rngSV[0]=_uint64(rngSV[0]+initstate)
  _random_r(rngSV)

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def srandom_r(unsigned long [:] rngSV, unsigned long initstate, unsigned long initseq):
  _srandom_r(rngSV,initstate,initseq)

cdef unsigned long bigC=6364136223846793005

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef unsigned int _random_r(unsigned long [:] rngSV):
  cdef unsigned long oldstate=rngSV[0]
  global bigC
  rngSV[0]=oldstate*bigC+rngSV[1]
  #rngSV[0]=_uint64(oldstate*bigC+rngSV[1])
  cdef unsigned int xorshifted=_uint32(((oldstate>>18)^oldstate)>>27)
  cdef unsigned int rot=_uint32(oldstate>>59)
  return _uint32((xorshifted>>rot) | (xorshifted<<((-rot)&31)))

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def random_r(unsigned long [:] rngSV):
  return _random_r(rngSV)

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef unsigned int _boundedrand_r(unsigned long [:] rngSV, unsigned int bound):
  cdef unsigned int threshold=-bound % bound
  #cdef unsigned int threshold=_uint32(-bound % bound)
  cdef unsigned int r
  while(True):
    r=_random_r(rngSV)
    if r>=threshold:
      return r % bound
      #return _uint32(r % bound)

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def boundedrand_r(unsigned long [:] rngSV, unsigned int bound):
  return _boundedrand_r(rngSV, bound)

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
cdef double _unitInterval_r(unsigned long [:] rngSV):
  cdef:
    unsigned int i
    unsigned int m=0xffffffff
    double f
  i=_random_r(rngSV)
  f=float(i)/m
  return f

@cython.cdivision(True)
@cython.nonecheck(False)
@cython.wraparound(False)
@cython.boundscheck(False)
def unitInterval_r(unsigned long [:] rngSV):
  return _unitInterval_r(rngSV)

cdef _random_r_arr(unsigned long [:] rngSV, int N):
  arr=np.zeros(N,dtype=np.uint32)
  cdef unsigned int [:] arrV = arr
  cdef int i
  for i in range(0,N):
    arrV[i]=_random_r(rngSV)
  return arr

def random_r_arr(unsigned long [:] rngSV, int N):
  return _random_r_arr(rngSV,N)

cdef _boundedrand_r_arr(unsigned long [:] rngSV, unsigned int bound, int N):
  arr=np.zeros(N,dtype=np.uint32)
  cdef unsigned int [:] arrV = arr
  cdef int i
  for i in range(0,N):
    arrV[i]=_boundedrand_r(rngSV,bound)
  return arr

def boundedrand_r_arr(unsigned long [:] rngSV, unsigned int bound, int N):
  return _boundedrand_r_arr(rngSV,bound,N)

cdef _unique_random_r_arr(unsigned long [:] rngSV, int N):
  cdef unsigned int [:] arrV,arr2V
  cdef unsigned int val
  cdef int i=0,j,l,larr
  arr=_random_r_arr(rngSV,N)
  arr=np.unique(arr)
  larr=arr.shape[0]
  l=N-larr
  arr2=np.zeros(l,dtype=np.uint32)
  arrV=arr
  arr2V=arr2
  while i<l:
    val=_random_r(rngSV)
    j=np.searchsorted(arr,val)
    if j==larr or arrV[j]!=val:
      arr2V[i]=val
      i+=1
  return np.union1d(arr,arr2)

def unique_random_r_arr(unsigned long [:] rngSV, int N):
  return _unique_random_r_arr(rngSV,N)

cdef _unique_boundedrand_r_arr(unsigned long [:] rngSV, unsigned int bound, int N):
  cdef unsigned int [:] arrV,arr2V
  cdef unsigned int val
  cdef int i=0,j,l,larr
  arr=_boundedrand_r_arr(rngSV,bound,N)
  arr=np.unique(arr)
  larr=arr.shape[0]
  l=N-larr
  arr2=np.zeros(l,dtype=np.uint32)
  arrV=arr
  arr2V=arr2
  while i<l:
    val=_boundedrand_r(rngSV,bound)
    j=np.searchsorted(arr,val)
    if j==larr or arrV[j]!=val:
      arr2V[i]=val
      i+=1
  return np.union1d(arr,arr2)

def unique_boundedrand_r_arr(unsigned long [:] rngSV, unsigned int bound, int N):
  return _unique_boundedrand_r_arr(rngSV,bound,N)

if __name__ == '__main__':
  #make a generator
  rngS=rng()
  #must seed
  srandom_r(rngS,42,52)
  for i in range(16):
    print(random_r(rngS))
