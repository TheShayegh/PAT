from Library.NaiveBayes import NBLM
from Library.Activation import Exp2Activation
import numpy as np
from tqdm import tqdm
from itertools import islice
from os import mkdir
from Library.Utils import save,load

class NB3 :
  def __init__(self, root, layers, data, actual_labels_freq=None, log=True,
               activation_class=Exp2Activation, activation=None, NB_class=NBLM) :
    self.root = root
    if type(activation_class) != type(list()) :
      activation_class = [activation_class for i in range(len(layers)-1)]
    self.activation_class = activation_class
    if type(NB_class) != type(list()) :
      NB_class = [NB_class for i in range(len(layers)-1)]
    self.NB_class = NB_class
    self.is_leaf = (len(layers) == 2)
    self.layer = layers[0]
    self.learning_data = layers[-1]
    self.dynasty_layers = layers[:-1]
    labels = actual_labels_freq[self.layer]   \
          if type(actual_labels_freq) != type(None) \
          else None

    self.model = self.NB_class[0]()
    self.model.train(data[[self.layer, self.learning_data]].values, labels=labels)

    self.activations = activation
    if type(activation) == type(None) :
      self.activations = [activation_cls() for activation_cls in self.activation_class]
    self.activation = self.activations[0]
    self.activation.update(self.model, data[self.learning_data], data[layers[0]].tolist())

    if not self.is_leaf :
      self.be_a_parent(layers, data, actual_labels_freq, log=log)
    
    if type(activation) == type(None) :
      for this_activation in self.activations :
        this_activation.process()

  def be_a_parent(self, layers, data, actual_labels_freq, log=False) :
      self.children = {}
      self.numof_grandchilds = {}
      tqd = tqdm(np.unique(data[self.layer].values)) if log else np.unique(data[self.layer].values)
      for child in tqd :
        sub_data = data[data[self.layer].values==child]
        sub_actual_labels_freq = actual_labels_freq[actual_labels_freq[self.layer].values==child] \
                                 if type(actual_labels_freq) != type(None) else None
        self.children[child] = NB3(root   = child,
                                   layers = layers[1:],
                                   data   = sub_data,
                                   actual_labels_freq = sub_actual_labels_freq,
                                   log    = False,
                                   activation_class = self.activation_class[1:],
                                   NB_class = self.NB_class[1:],
                                   activation = self.activations[1:],
                                   )
        self.numof_grandchilds[child] = len(self.children[child].model._labels)
        
  def parts_count(self, children_batch_size=50) :
    return 2 + len(self.children)//children_batch_size

  def json_object(self, keep_activations=True, log=True, part=-1, children_batch_size=50) :
    info = dict()
    info['layers'] = self.dynasty_layers+[self.learning_data]

    if part == -1 or part == 0 :
      info['root'] = self.root
      info['model'] = self.model.json_object()
      if keep_activations :
        info['_activations'] = list()
        for ac in self.activations :
          info['_activations'].append(ac.json_object())

    if part == -1 or part > 0 :
      if not self.is_leaf :
        _range = (0, len(self.children))
        if part > 0 :
          _range = (children_batch_size*(part-1), children_batch_size*(part))
        info['numof_grandchilds'] = dict(islice(self.numof_grandchilds.items(), _range[0], _range[1]))
        info['children'] = dict()
        tqd = islice(self.children.items(), _range[0], _range[1])
        tqd = tqdm(tqd, total=min(_range[1]-_range[0], len(self.children)-_range[0])) if log else tqd
        for child_name,child in tqd :
          info['children'][child_name] = child.json_object(keep_activations=False, log=False)
        del tqd
    return info

  def loot_children(self, other) :
    self.children.update(other.children)
    self.numof_grandchilds.update(other.numof_grandchilds)
      
  @classmethod
  def loads(cls, info, activation_class=Exp2Activation, NB_class=NBLM, activations=None, log=True, part=-1) :
    c = cls.__new__(cls)
    layers = info['layers']
    if type(activation_class) != type(list()) :
      activation_class = [activation_class for i in range(len(layers)-1)]
    if type(NB_class) != type(list()) :
      NB_class = [NB_class for i in range(len(layers)-1)]

    if part == -1 or part == 0 :
      c.root = info['root']
      c.activation_class = activation_class
      c.NB_class = NB_class
      c.is_leaf = (len(layers) == 2)
      c.layer = layers[0]
      c.learning_data = layers[-1]
      c.dynasty_layers = layers[:-1]
      c.model = c.NB_class[0].loads(info['model'])
      c.activations = activations
      if type(activations) == type(None) :
        c.activations = list()
        for i,ac in enumerate(info['_activations']) :
          c.activations.append(activation_class[i].loads(ac))
        activations = c.activations
      c.activation = c.activations[0]
      if not c.is_leaf :
        c.numof_grandchilds = dict()
        c.children = dict()

    if (part == -1 and not c.is_leaf) or part > 0 :
      c.numof_grandchilds = info['numof_grandchilds']
      c.children = {}
      tqd = info['children'].items()
      tqd = tqdm(tqd) if log else tqd
      for child_name,child in tqd :
        c.children[child_name] = cls.loads(child,
                                          activation_class = activation_class[1:],
                                          NB_class         = NB_class[1:],
                                          activations      = activations[1:],
                                          log              = False)
      del tqd
    del info
    return c
    


  def post_process(self, data) :
    output = []
    for label,(prob,status) in data.items() :
      if status == 'X' :
        continue
      if status == 'T' :
        output.append({'probability': prob, self.layer: label})
      else :
        subresult = status[0]
        for d in subresult :
          d['probability'] = self.activation.combine_with_children(prob,d['probability'])
          d[self.layer] = label
          output.append(d)
    return output
        
  def run(self, address, sort_output=True, log=False) :
    probs = self.model[address]
    probs = probs._prob_dict
    probs = self.activation.get_dist(probs)
    
    result = {}
    
    if self.is_leaf :
      for label, prob in probs.items() :
        if prob == 0 :
          result[label] = (prob, 'X')
        else :
          result[label] = (prob, 'T')
        
    else :
      best_label, max_prob = max(probs.items(), key=lambda x: x[1])
      noc = self.numof_grandchilds.get(best_label, 0)
      threshold = self.activation.brothers_threshold(max_prob, noc)

      for label, prob in probs.items() :
        if prob < threshold or label not in self.children :
          result[label] = (prob, 'X')
        else :
          result[label] = (prob, self.children[label].run(address, sort_output=False))
    
    temp = sorted(self.post_process(result), key=lambda x: x['probability'], reverse=True) \
           if sort_output else self.post_process(result)
    result = (temp, result)
    
    if log :
      NB3.print(result)

    return result

  def __getitem__(self, address) :
    return self.run(address)

  def test(self, test_data, neighbor_avenues=None, delimiter='-', top=10, log=True) :
    result = {i: {l: 0 for l in self.dynasty_layers} for i in range(top)}
    if type(neighbor_avenues) != type(None) :
      for d in result.values() :
        d.update({"neighbor": 0})
    tqd = test_data.iterrows()
    tqd = tqdm(tqd, total=test_data.shape[0]) if log else tqd
    for rindex,row in tqd :
        label = {l: row[l] for l in self.dynasty_layers}
        output,_ = self[row[self.learning_data]]
        for li,l in enumerate(self.dynasty_layers) :
            for i in range(min(top, len(output))) :
                if all([output[i][ll] == label[ll] for ll in self.dynasty_layers[:li+1]]) :
                    result[i][l] += 1
                    break
        if type(neighbor_avenues) == type(None) :
          continue
        for i in range(min(top, len(output))) :
            prediction = delimiter.join([output[i][ll] for ll in self.dynasty_layers])
            true_label = delimiter.join([label[ll] for ll in self.dynasty_layers])
            if prediction in neighbor_avenues[true_label] :
                result[i]["neighbor"] += 1
                break
    return result

  def save(self, path, children_batch_size=50, log=True) :
    if path[-1] != '/' :
      path += '/'
    try : mkdir(path)
    except : pass
    tqd = range(self.parts_count(children_batch_size=children_batch_size))
    tqd = tqdm(tqd) if log else tqd
    for i in tqd :
      save(self, path+'part'+str(i), part=i, children_batch_size=children_batch_size, log=False)

  @classmethod
  def load(cls, path, parts_count=1, activation_class=Exp2Activation, NB_class=NBLM, activations=None, log=True) :
    if path[-1] != '/' :
      path += '/'
    c = load(cls, path+'part'+str(0),
              activation_class=activation_class, NB_class=NB_class,
              activations=activations, log=False, part=0)
    tqd = range(1,parts_count)
    tqd = tqdm(tqd) if log else tqd
    for i in tqd :
      c.loot_children(load(NB3, path+'part'+str(i),
                            activation_class=activation_class, NB_class=NB_class,
                            activations=c.activations, log=False, part=1))
    return c
    
    





class TestModel :
  def __init__(self, tested_data, layers) :
    self.dynasty_layers = layers[:-1]
    self.learning_data = layers[-1]
    self.tested_data = tested_data # dict: {text -> NB3 output[0]}

  def __getitem__(self, address) :
    return self.tested_data[address],None
    
  def test(self, **args) :
    return NB3.test(self, **args)