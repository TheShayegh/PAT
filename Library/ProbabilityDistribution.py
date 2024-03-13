from nltk.probability import ProbDistI,FreqDist,LaplaceProbDist
from scipy.stats import gaussian_kde
import numpy as np
import dill


class BernoulliProbDist(ProbDistI) :
  def __init__(self, featureset) :
    self.exist = featureset["exist"]
    self.n_samples = featureset["n_samples"]
    self.p = (self.exist)/(self.n_samples)
    self.q = 1-self.p

  def freqdist(self) :
    _freqdist = FreqDist()
    _freqdist[True] = self.exist
    _freqdist[False] = self.n_samples-self.exist
    return _freqdist

  def prob(self, p=True) :
    return self.p if p else self.q

  def max(self) :
    return self.freqdist().max()

  def samples(self) :
    return self.freqdist().keys()

  def discount(self) :
    raise NotImplementedError()

  def __repr__(self) :
    return f"<BernoulliProbDist with p={self.p}, samples count={self.n_samples}>"

  def json_object(self) :
    return self.__dict__.copy()
    
  @classmethod
  def loads(cls, j) :
    c = cls.__new__(cls)
    for attr,val in j.items() :
      setattr(c, attr, val)
    return c
    
 
    
    
    
class DumpableLaplaceProbDist(LaplaceProbDist) :
  def __init__(self, freqdist) :
      super().__init__(freqdist)

  def json_object(self) :
    info = self.__dict__.copy()
    info['_freqdist'] = dict(info['_freqdist'])
    return info
    
  @classmethod
  def loads(cls, j) :
    c = cls.__new__(cls)
    for attr,val in j.items() :
      setattr(c, attr, val)
    c._freqdist = FreqDist(c._freqdist)
    return c
            





class PositionProbDist(ProbDistI) :
  def __init__(self, featureset, bw_method=0.25) :
    positions = featureset["positions"]
    self.exist = len(positions)
    self.n_samples = featureset["n_samples"]
    self.p = (self.exist)/(self.n_samples)
    self.q = 1-self.p
    if self.exist > 0 :
      self.gkde = gaussian_kde([0,1]+positions, bw_method=bw_method)
      scale = 1/(sum([self.gkde((i+.5)/10) for i in range(10)])/10)
      self.scale = float(scale[0])
    del positions

  def freqdist(self) :
    _freqdist = FreqDist()
    if self.exist > 0 :
      for i in range(10) :
        _freqdist[(i+.5)/10] = int(self.gkde((i+.5)/10) * self.scale * self.exist)
    _freqdist[None] = self.n_samples-self.exist
    return _freqdist

  def prob(self, sample, left=None, right=None) :
    if sample == None :
      return self.q
    if self.exist > 0 :
      if left == None or right == None :
        return (self.gkde(sample) * self.scale * self.p)
      else :
        return (
            (self.gkde(left)+
             2*self.gkde(sample)+
             self.gkde(right)
            )/4 * self.scale * self.p)
    return self.p

  def max(self) :
    return self.freqdist().max()

  def samples(self) :
    return self.freqdist().keys()

  def discount(self) :
    raise NotImplementedError()

  def __repr__(self) :
    return f"<PositionProbDist with p={self.p}, samples count={self.n_samples}>"

  def json_object(self) :
    info = self.__dict__.copy()
    info['gkde'] = dill.dumps(self.gkde)
    return info

  @classmethod
  def loads(cls, j) :
    c = cls.__new__(cls)
    for attr,val in j.items() :
      setattr(c, attr, val)
    c.gkde = dill.loads(c.gkde)
    return c