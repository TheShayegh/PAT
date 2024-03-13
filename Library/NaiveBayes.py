from nltk.classify.naivebayes import NaiveBayesClassifier as nltkNB
from Library.ProbabilityDistribution import BernoulliProbDist,PositionProbDist,DumpableLaplaceProbDist
from nltk.probability import FreqDist,DictionaryProbDist
from collections import defaultdict
from nltk import sum_logs
import numpy as np
import math
import pickle





class NBLM(nltkNB) : # Naive Bayes Language Model
  def __init__(self, smooth_factor=0, idf_power=0, featuresProbEstimator=BernoulliProbDist, labelProbEstimator=DumpableLaplaceProbDist) :
    self.smooth_factor = smooth_factor
    self.idf_power = idf_power
    self.idf = lambda fname: 1
    if self.idf_power != 0 :
      self.idf = lambda fname: math.log(self._feature_freqdist_sum/self._feature_freqdist[fname], 2)**self.idf_power
    self.featuresProbEstimator = featuresProbEstimator
    self.labelProbEstimator = labelProbEstimator

  def tokenize(text) :
    return text.split()

  def run(self, text) :
    if type(text) == type('') :
      return self.prob_classify(text)
    else :
      return self.prob_classify_many(text)

  def __getitem__(self, text) :
    return self.run(text)

  def featureset_probs(self, featureset) :
    logprob = {}
    for label in self._labels:
      logprob[label] = self._label_probdist.logprob(label)
      for fname in featureset:
        p = self.smooth_factor
        if (label, fname) in self._feature_probdist:
          feature_probs = self._feature_probdist[label, fname]
          p += feature_probs.prob()
        logprob[label] += math.log(p, 2)*self.idf(fname) if p > 0 else sum_logs([])
    return logprob

  def featureset_preparation(self, featureset) :
    length = len(featureset)
    for index in range(length-1,0-1,-1):
      fname = featureset[index]
      for label in self._labels:
        if (label, fname) in self._feature_probdist:
          break
      else:
        del featureset[index]

  def prob_classify(self, featureset):
    featureset = NBLM.tokenize(featureset)
    self.featureset_preparation(featureset)
    logprob = self.featureset_probs(featureset)
    return DictionaryProbDist(logprob, normalize=True, log=True)

  def train(self, labeled_texts, labels=None):
    idf_enable = (self.idf_power != 0)
    label_freqdist = FreqDist()
    features = defaultdict(lambda: {"exist": 0, "n_samples": 0})
    if idf_enable :
      feature_freqdist = FreqDist()

    for label,featureset in labeled_texts:
      label_freqdist[label] += 1
      for fname in NBLM.tokenize(featureset) :
        features[label, fname]["exist"] += 1
        if idf_enable :
          feature_freqdist[fname] += 1
    
    feature_probdist = {}
    for ((label, fname), featureset) in features.items():
      featureset["n_samples"] = label_freqdist[label]
      probdist = self.featuresProbEstimator(featureset)
      feature_probdist[label, fname] = probdist
    del features

    if type(labels) != type(None) :
      label_freqdist = FreqDist(label for label in labels)
    label_probdist = self.labelProbEstimator(label_freqdist)
    
    self._label_probdist = label_probdist
    self._feature_probdist = feature_probdist
    self._labels = list(label_probdist.samples())
    if idf_enable :
      self._feature_freqdist = feature_freqdist
      self._feature_freqdist_sum = sum(list(feature_freqdist.values()))

  def json_object(self) :
    info = dict()
    info['smooth_factor'] = self.smooth_factor
    info['idf_power'] = self.idf_power
    if (self.idf_power != 0) :
      info['_feature_freqdist'] = dict(self._feature_freqdist)
      info['_feature_freqdist_sum'] = self._feature_freqdist_sum
    info['_label_probdist'] = self._label_probdist.json_object()
    info['feature_probdist'] = dict()
    for (label,fname),feature_probdist in self._feature_probdist.items() :
      info['feature_probdist'][pickle.dumps([label,fname])] = feature_probdist.json_object()
    return info

  @classmethod
  def loads(cls, info) :
    c = cls()
    for attr,val in info.items() :
      setattr(c, attr, val)
    if (c.idf_power != 0) :
      c._feature_freqdist = FreqDist(c._feature_freqdist)
    c._label_probdist = c.labelProbEstimator.loads(c._label_probdist)
    c._labels = list(c._label_probdist.samples())
    c._feature_probdist = {}
    for label_fname,feature_probdist_path in c.feature_probdist.items() :
      label,fname = pickle.loads(label_fname)
      c._feature_probdist[label,fname] = c.featuresProbEstimator.loads(c.feature_probdist[label_fname] )
    del c.feature_probdist
    return c
    
    






class PNBLM(NBLM) : # Positional Naive Bayes Language Model
  def __init__(self, smooth_factor=0, idf_power=0, bw_method=0.25, featuresProbEstimator=PositionProbDist, labelProbEstimator=DumpableLaplaceProbDist) :
    self.bw_method = bw_method
    super().__init__(smooth_factor=smooth_factor, idf_power=idf_power, featuresProbEstimator=featuresProbEstimator, labelProbEstimator=labelProbEstimator)
  
  def featureset_probs(self, featureset) :
    length = len(featureset)
    logprob = {}
    for label in self._labels:
      logprob[label] = self._label_probdist.logprob(label)
      for index,fname in enumerate(featureset):
        p = self.smooth_factor
        if (label, fname) in self._feature_probdist:
          feature_probs = self._feature_probdist[label, fname]
          p += feature_probs.prob((index+.5)/length)
        logprob[label] += math.log(p, 2)*self.idf(fname) if p > 0 else sum_logs([])
    return logprob

  def train(self, labeled_texts, labels=None):
    idf_enable = (self.idf_power != 0)
    label_freqdist = FreqDist()
    features = defaultdict(lambda: {"positions": list(), "n_samples": 0})
    if idf_enable :
      feature_freqdist = FreqDist()

    for label,text in labeled_texts:
      featureset = NBLM.tokenize(text)
      label_freqdist[label] += 1
      length = len(featureset)
      for index,fname in enumerate(featureset) :
        features[label, fname]["positions"].append((index+.5)/length)
        if idf_enable :
          feature_freqdist[fname] += 1
    
    feature_probdist = {}
    for ((label, fname), featureset) in features.items():
      featureset["n_samples"] = label_freqdist[label]
      probdist = self.featuresProbEstimator(featureset, bw_method=self.bw_method)
      feature_probdist[label, fname] = probdist
    del features

    if type(labels) != type(None) :
      label_freqdist = FreqDist(label for label in labels)
    label_probdist = self.labelProbEstimator(label_freqdist)
    
    self._label_probdist = label_probdist
    self._feature_probdist = feature_probdist
    self._labels = list(label_probdist.samples())
    if idf_enable :
      self._feature_freqdist = feature_freqdist
      self._feature_freqdist_sum = sum(list(feature_freqdist.values()))