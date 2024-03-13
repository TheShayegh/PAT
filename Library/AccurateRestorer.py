import re
import json

class AccurateRestorer :
  def __init__(self, layers,
               urban_hierarchy,
               preprocessor=lambda x: x,
               prob_keyword='probability',
               label_cond_thresholds=1,
               plateno_cond_thresholds=1) :
    self.preprocessor = preprocessor
    self.layers = layers
    self.prob_keyword = prob_keyword
    if type(label_cond_thresholds) != type(dict()) :
      label_cond_thresholds = {l: label_cond_thresholds for l in self.layers}
    self.label_cond_thresholds = label_cond_thresholds
    self.plateno_cond_thresholds = plateno_cond_thresholds
    self.avenue_elements = json.load(open("./Library/AvenueElements.json"))
    self.urban_hierarchy = urban_hierarchy

  def exists_in_address(self, key, address) :
    keys = re.split('|'.join(self.avenue_elements),key)
    return all([k in address for k in keys])

  def apply_label_cond(self, prediction, address, layer) :
    if len(prediction) == 1 :
      return prediction
    threshold = self.label_cond_thresholds[layer]
    for i,p in enumerate(prediction) :
      if p[self.prob_keyword] < threshold :
        break
      if self.exists_in_address(self.preprocessor.run(p[layer]), address) :
        return [p]+prediction[:i]+prediction[i+1:]
    return prediction

  def apply_plateno_cond(self, prediction, plateno, keep_layers=[], check_layers=None, ignore_prob=None, pushdown_edge=5) :
    if len(prediction) == 1 :
      return prediction
    if type(plateno) == type(None) :
      return prediction
    ignore_prob = self.plateno_cond_thresholds
    if type(check_layers) == type(None) :
      check_layers = self.layers
    p0 = prediction[0]
    for i,p in enumerate(prediction) :
      if any(p[l] != p0[l] for l in keep_layers) :
        continue
      if p[self.prob_keyword] < ignore_prob :
        break
      sp = self.urban_hierarchy
      for l in check_layers :
        sp = sp[p[l]]
      if str(plateno) in sp :
        if i == 0 :
          return prediction
        if i < pushdown_edge :
          return [p]+prediction[:i]+prediction[i+1:]
        else :
          ps = [p]+prediction[1:i]+prediction[i+1:]
          ps = ps[:pushdown_edge]+[p0]+ps[pushdown_edge:]
          return ps
    return prediction