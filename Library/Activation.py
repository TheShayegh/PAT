from math import ceil
import numpy as np


class ZeroActivation :
  def __init__(self, ) :
    pass
    
  def get_dist(self, dist) :
    return dist
    
  def update(self, nb_model, X, Y) :
    pass

  def process(self) :
    pass
    
  def combine_with_children(self, a, b) :
    return a+b

  def brothers_threshold(self, m, n) :
    return 2*m

  def json_object(self) :
    return ""

  @classmethod
  def loads(cls, j) :
    return cls()
    
    
    
    
    
    
    
class Exp2Activation(ZeroActivation) :
  def __init__(self, ) :
    pass
    
  def get_dist(self, dist) :
    return {k: 2**v for k,v in dist.items()}
    
  def combine_with_children(self, a, b) :
    return a*b

  def brothers_threshold(self, m, n) :
    return m / n if n > 0 else 1
    
    
    
    
    
    

class Score2Prob :
  eps = float(np.finfo(float).eps)

  def __init__(self, finesse=0.001, window=(-0,1), input_transform=lambda x: x) :
    self.finesse = finesse
    self.window = window
    self.input_transform = input_transform
    self.length = int((1+Score2Prob.eps)//self.finesse)+1
    self.true__buckets = [0]*self.length
    self.count_buckets = [0]*self.length

  def add(self, data) :
    for p,t in data :
      p = self.input_transform(p)
      index = int((p+Score2Prob.eps)//self.finesse)
      self.true__buckets[index] += t
      self.count_buckets[index] += 1

  def process(self) :
    l,r = self.window
    l = -l
    true__buckets = [sum(([0]*(l)+self.true__buckets)[i-l:i+r]) for i in range(l,self.length+l)]
    count_buckets = [sum(([0]*(l)+self.count_buckets)[i-l:i+r]) for i in range(l,self.length+l)]
    r = ceil(-np.log(self.finesse)/np.log(10))
    self.prob_buckets = [round(t/c, r) if c>0 else -1 for t,c in zip(true__buckets, count_buckets)]
    for i,p in enumerate(self.prob_buckets) :
      if p == -1 :
        self.prob_buckets[i] = self.prob_buckets[i-1]
    
  def get(self, p) :
    p = self.input_transform(p)
    index = int((p+Score2Prob.eps)//self.finesse)
    return self.prob_buckets[index]

  def change_window(self, window) :
    self.window = window

  def json_object(self) :
    return self.__dict__.copy()
    
  @classmethod
  def loads(cls, j) :
    c = cls()
    for attr,val in j.items() :
      setattr(c, attr, val)
    return c







class Score2ProbActivation(Exp2Activation) :
  def __init__(self, finesse=0.001, window=(-5,6), input_transform=lambda x: 2**x) :
    self.score2Prob = Score2Prob(finesse=finesse, window=window, input_transform=input_transform)
    
  def update_with_output(self, probs, labels) :
    for prob,label in zip(probs, labels) :
      P = list(prob._prob_dict.values())
      Y = (np.array(list(prob._prob_dict.keys())) == label).astype(int).tolist()
      self.score2Prob.add(zip(P,Y))
    
  def update(self, nb_model, X, Y) :
    probs = nb_model.run(X)
    self.update_with_output(probs, Y)

  def process(self) :
    self.score2Prob.process()

  def get_dist(self, dist) :
    output = dist.copy()
    for k in output :
      output[k] = self.score2Prob.get(output[k])
    return output

  def json_object(self) :
    return self.score2Prob.json_object()

  @classmethod
  def loads(cls, j) :
    c = cls()
    c.score2Prob = Score2Prob.loads(j)
    return c