import os,sys
import pandas as pd
import numpy as np
import pickle

jnb_output = sys.stdout
def blockPrint():
  sys.stdout = open(os.devnull, 'w')
def enablePrint():
  sys.stdout = jnb_output

def replace_all(s, words, source_context=' {} ', target_context=' {} '):
  for k,v in words:
    f = source_context.format(k)
    t = target_context.format(v)
    if f!=t :
      s = s.replace(f, t)
  return s
  
def large_unique(arr, max_size=10000000) :
  df = pd.DataFrame(index=[],data={'c':[]})
  for i in range(0,len(arr),max_size) :
    temp = arr[i:i+max_size]
    u,c = np.unique(temp, return_counts=True)
    df = df.add(pd.DataFrame(index=u,data={'c':c}), fill_value=0)
  return df.index.values, df.c.values.astype(int)

softmax = lambda x: np.exp(x)/np.exp(x).sum()

def cut_till(text, word) :
  if type(word) == type('') :
    x = text.find(word)
    if x < 0 :
      return text
    return text[:x]
  else :
    t = text
    for w in word :
      t = cut_till(t, w)
    return t

def select_unique_test_data(data, test_size, _range, unique_on) :
  temp = data[_range[0]:_range[1]].drop_duplicates(unique_on)
  test_index = np.zeros(temp.shape[0]).astype(bool)
  test_index[np.random.choice(temp.shape[0], test_size, replace=False)] = True
  test = temp[test_index]
  train = data[~np.isin(data[unique_on].values, test[unique_on].values)]
  return test,train

def select_test_data(data, test_size, _range=None, unique_on=None) :
  data_size = data.shape[0]
  if _range == None :
      _range = (0,data_size)
  if type(unique_on) != type(None) :
    return select_unique_test_data(data, test_size, _range, unique_on)
  test_index = np.zeros(data_size).astype(bool)
  test_index[np.random.choice(np.arange(_range[0],_range[1]), test_size, replace=False)] = True
  return data[test_index], data[~test_index]
  
  
def sqrt_unique(data, col) :
  subdfs = list()
  for _,subdf in data.groupby(col) :
    subdfs.append(subdf[:int(subdf.shape[0]**0.5)])
  return data[0:0].append(subdfs)
  
  
 
def save(obj, path, to_json_func="json_object", **keywords) :
  with open(path+'.pkl', 'wb') as f :
    pickle.dump(getattr(obj, to_json_func)(**keywords), f)

def load(cls, path, read_json_func="loads", **keywords) :
  with open(path+'.pkl', 'rb') as f :
    return getattr(cls, read_json_func)(pickle.load(f), **keywords)
    
    
def flatten(listoflist) :
  output = []
  for l in listoflist :
    output += l
  return output