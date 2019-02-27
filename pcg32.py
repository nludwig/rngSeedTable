#code adapted by Snack-X
#from code & theory due to
#M.E. O'Neill: see
#www.pcg-random.org

def _uint32(n):
  return n & 0xffffffff

def _uint64(n):
  return n & 0xffffffffffffffff

def rng(state=0, inc=0):
  return {
    'state': _uint64(state),
    'inc': _uint64(inc)
  }

_pcg32_global = rng(0x853c49e6748fea9b, 0xda3e39cb94b95bdb)

def srandom_r(rng, initstate, initseq):
  initstate = _uint64(initstate)
  initseq = _uint64(initseq)

  rng['state'] = 0
  rng['inc'] = _uint64(initseq << 1) | 1
  random_r(rng)
  rng['state'] = _uint64(rng['state'] + initstate)
  random_r(rng)

def srandom(seed, seq):
  srandom_r(_pcg32_global, seed, seq)

def random_r(rng):
  oldstate = rng['state']
  rng['state'] = _uint64(oldstate * 6364136223846793005 + rng['inc'])
  xorshifted = _uint32(((oldstate >> 18) ^ oldstate) >> 27)
  rot = _uint32(oldstate >> 59)
  return _uint32((xorshifted >> rot) | (xorshifted << ((-rot) & 31)))

def random():
  return random_r(_pcg32_global)

def boundedrand_r(rng, bound):
  threshold = _uint32(-bound % bound)
  while(True):
    r = random_r(rng)
    if r >= threshold:
      return _uint32(r % bound)

def unitInterval_r(rng):
  i=random_r(rng)
  f=i/0xffffffff
  return f

def main():
  #make a generator
  r = rng()
  srandom_r(r, 42, 52) #seed
  for i in range(16):
    print(random_r(r))

  import sys
  for i in range(0,100):
    print(unitInterval_r(r),file=sys.stderr)

  #or use global
  srandom(42, 52) #seed
  for i in range(16):
    print(random())

if __name__ == '__main__':
  main()
