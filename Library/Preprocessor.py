import re
from hazm import Normalizer
import numpy as np
from tqdm import tqdm 
import json
from Library.Utils import replace_all,large_unique
from Library.Utils import save,load
from math import log,inf,sqrt
from sys import exit













class Vocabulary :
  def __init__(self, words=None) :
    if words == None :
      words = dict()
    self.words = words
    self.warning = False

  def new_words(self, words) : #words is something like list
    self.words.update({k:0 for k in set(words)-set(self.words.keys())})

  def just_frequents(self) :
    return Vocabulary({w:c for w,c in self.words.items() if c > 0})

  def get(self, word) : #word is string
    return self.words.get(word, -1)
    
  def get_batch(self, words, include_misseds=False) :#words is something like list
    output = {w: self.get(w) for w in words}
    if include_misseds :
      return output
    output = {w:c for w,c in output.items() if c >= 0}
    return output

  def add_word(self, word, count=1, new=False) :
    if new :
      self.new_words([word])
    c = self.get(word)
    if c < 0 and self.warning :
      print("WARNING:",'"{}"'.format(word),"that you wish to update it's count, not exists in vocabulary!")
    if c >= 0 :
      self.words[word] = c + count 

  def add_words(self, words, new=False) : #words is dict
    if new :
      self.new_words(list(words.keys()))
    for word,count in words.items() :
      self.add_word(word,count)

  def add_text(self, text, new=False) : #text is string
    words = large_unique(text.split())
    words = {w:c for w,c in zip(words[0].tolist(),words[1].tolist())}
    self.add_words(words, new=new)

  def __getitem__(self, word) :
    return self.get(word)

  def __str__(self) :
    return str(self.words)
    
  def __contains__(self, word) :
    return word in self.words

  def json_object(self) :
    return self.__dict__.copy()
    
  def copy(self) :
    new = Vocabulary(self.words.copy())
    new.warning = self.warning
    return new

  @classmethod
  def loads(cls, j) :
    c = cls()
    for attr,val in j.items() :
      setattr(c, attr, val)
    return c



























class SpaceRemover :
  def __init__(self, vocab) :
    self.vocab = vocab
    self.max_len = max(list(map(len, self.vocab.words.keys())))
    self.history = dict()

  def clear_history(self) :
    del self.history
    self.history = dict()

  def run(self, text) :
    h = self.history.get(text, -1)
    if h != -1 :
      h = h if h[0] != 0 else (text,h[1],h[2],h[3],h[4])
      return h
    h = self.remove_wrong_spaces(text)
    self.history[text] = h if text != h[0] else (0,h[1],h[2],h[3],h[4])
    return h

  def concat(self, right, left) :
    a = min(right[4],left[4])
    if a < 0 :
      a = max(right[4],left[4])
    return right[0]+' '+left[0], right[1]+left[1], right[2]+left[2], right[3]+left[3], a
    
  def is_better(self, new, old) :
    if new[1] < old[1] :
      return True
    if new[4] >= old[4] :
      return (new[2], new[3]) < (old[2], old[3])
    return False

  def remove_wrong_spaces(self, text) :
    n = len(text)
    if n <= self.max_len :
      c = self.vocab[text]
      if c >= 0 :
        return text,0,1,-n**2,c
    if ' ' not in text :
      return text,1,1,-n**2,-1
    t = text.replace(' ','')
    n = len(t)
    ts = None
    if n < self.max_len :
      c = self.vocab[t]
      if c >= 0 :
        ts = t,0,1,-n**2,c
    t = text.split()
    n = len(t)
    if ts == None :
      ts = text,n,n,-n**2,-1
    for i in range(1,n) :
      fright = self.run(' '.join(t[:i]))
      fleft  = self.run(' '.join(t[i:]))
      newf = self.concat(fright, fleft)
      if self.is_better(newf, ts) :
        ts = newf
    return ts
  
  def __getitem__(self, text) :
    return self.run(text)[0]

  def json_object(self) :
    j = self.__dict__.copy()
    j['vocab'] = self.vocab.json_object()
    return j

  @classmethod
  def loads(cls, j) :
    c = cls(Vocabulary.loads(j['vocab']))
    for attr,val in j.items() :
      if attr != 'vocab' :
        setattr(c, attr, val)
    return c
  
  def find_difference(self, a, b=None) :
    if b == None :
      b = self[a]
    a = a.split()
    b = b.split()
    aa = list()
    result = dict()
    j = 0
    for i in a :
      aa.append(i)
      str_a = ''.join(aa)
      if str_a == b[j] :
        if len(aa) > 1 :
          result[' '.join(aa)] = b[j]
        j += 1
        aa = list()
    return result























class SpaceInserter :
  def __init__(self, vocab) :
    self.vocab = vocab
    self.cal_max_scores()
    self.guide_history = dict()
    self.history = dict()

  def clear_history(self) :
    del self.guide_history
    del self.history
    self.guide_history = dict()
    self.history = dict()

  def cal_max_scores(self) :
    self.max_count = dict()
    for w,c in self.vocab.words.items() :
      l = len(w)
      self.max_count[l] = max(self.max_count.get(l,0), c)
    self.max_score = dict()
    for l in self.max_count :
      for i in range(int((l+1)/2)) :
        j = l-i
        isc = self.score(self.max_count.get(i,0),i)
        jsc = self.score(self.max_count.get(j,0),j)
        self.max_score[l] = max(self.max_score.get(l,0), self.sum_scores(isc,jsc))

  def score(self, wc, length=None) :
    if length==None :
      length = len(wc)
      wc = self.vocab[wc]
    return log(0.2*(wc+1)+1)*length

  def get_max_score(self, l) :
    return self.max_score.get(l,0)

  def sum_scores(self, a, b) :
    return sqrt(a**2+b**2)

  def enough_good(self, new, old) :
    return new[1] > old[1]

  def add_to_history(self, word, result, alpha) :
    if result[0] == '' :
      self.guide_history[word] = -alpha if alpha != 0 else -inf
    elif not result[0] == word :
      self.guide_history[word] = result[1]
      self.history[word] = result

  def run(self, w, alpha=None) :
    n = len(w)
    if alpha == None :
      alpha = 0
    h = self.guide_history.get(w, None)
    if h != None :
      if h < 0 :
        if h == -inf :
          h = 0
        if alpha > -h :
          return '',h
      else :
        if h == inf :
          return w,self.score(self.vocab[w],n)
        if alpha < h :
          return self.history[w]
        else :
          return '',h
    result = self.add_spaces(w, alpha)
    self.add_to_history(w, result, alpha)
    return result

  def add_spaces(self, w, alpha) :
    n = len(w)
    c = self.score(self.vocab[w],n)
    result = '',alpha
    fake_result = w,c
    if self.enough_good(fake_result, result) :
      result = fake_result
    candidates = []
    for i in range(1,n) :
      wright = w[:i]
      wleft  = w[i:]
      neww   = wright+' '+wleft
      cwrite = self.get_max_score(i)
      cleft  = self.get_max_score(n-i)
      newc = self.sum_scores(cwrite,cleft)
      fake_result = neww,newc
      if not self.enough_good(fake_result, result) :
        continue
      cwrite = self.score(self.vocab[wright],i)
      cleft  = self.score(self.vocab[wleft],n-i)
      newc = self.sum_scores(cwrite,cleft)
      if newc == 0 :
        continue
      fake_result = neww,newc
      if cwrite*cleft > 0 :
        if self.enough_good(fake_result, result) :
          result = fake_result
      if cwrite > 0 :
        candidates.append((wright,wleft,cwrite,cleft))
    candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
    for wright,wleft,cwrite,cleft in candidates :
      l = len(wleft)
      newc = max([self.sum_scores(self.get_max_score(i), self.get_max_score(l-i)) for i in range(int((l+1)/2))])
      newc = self.sum_scores(cwrite,newc)
      neww = wright+' '+wleft
      fake_result = neww,newc
      if not self.enough_good(fake_result, result) :
        continue
      subword,newc = self.run(wleft, alpha=cleft)
      if subword == '' :
        continue
      neww = wright+' '+subword
      newc = self.sum_scores(cwrite,newc)
      fake_result = neww,newc
      if self.enough_good(fake_result, result) :
        result = fake_result
    return result

  def __getitem__(self, text) :
    if ' ' in text :
      return ' '.join(list(map(lambda x: self[x], text.split())))
    return self.run(text)[0]

  def json_object(self) :
    j = self.__dict__.copy()
    j['vocab'] = self.vocab.json_object()
    return j

  @classmethod
  def loads(cls, j) :
    c = cls(Vocabulary.loads(j['vocab']))
    for attr,val in j.items() :
      if attr == 'vocab' :
        continue
      if attr in ['max_count','max_score'] :
        d = dict()
        for i,v in val.items() :
          d[int(i)] = v
        setattr(c, attr, d)
      else :
        setattr(c, attr, val)
    return c
  
  def find_difference(self, b, a=None) :
    if a == None :
      a = self[b]
    a = a.split()
    b = b.split()
    aa = list()
    result = dict()
    j = 0
    for i in a :
      aa.append(i)
      str_a = ''.join(aa)
      if str_a == b[j] :
        if len(aa) > 1 :
          result[b[j]] = ' '.join(aa)
        j += 1
        aa = list()
    return result
























class SpellChecker :
  def __init__(self, vocab) :
    self.vocab = vocab
    self.max_len = max(list(map(len, self.vocab.words.keys())))
    self.create_mcw()
    self.edit_functions = {
      'self_exist' : self.self_exist,
      'inserting' : self.inserting,
      'transposing' : self.transposing,
      'deleting' : self.deleting,
      'replacing' : self.replacing,
    }

  def create_mcw(self) :
    self.mcw = dict()
    for i in range(self.max_len) :
      self.mcw[i] = dict()
    for w in self.vocab.words :
      for i in range(len(w)) :
        wm = w[:i]+w[i+1:]
        if wm not in self.mcw[i] :
          self.mcw[i][wm] = list()
        self.mcw[i][wm].append(w)

  def mcw_find(self, word, index=None) :
    if index == None :
      output = list()
      for i in range(len(word)+1) :
        output += self.mcw_find(word, index=i)
      return np.unique(output).tolist()
    if index >= self.max_len :
      return list()
    return self.mcw[index].get(word, list()).copy()

  def self_exist(self, word) :
    return self.vocab.get(word)

  def inserting(self, word, index=None) :
    output = self.mcw_find(word, index=index)
    return self.vocab.get_batch(output)

  def replacing(self, word, index=None) :
    if index == None :
      output = dict()
      for i in range(len(word)) :
        output.update(self.replacing(word, index=i))
      return output
    output = self.mcw_find(word[:index]+word[index+1:], index=index)
    if word in output :
        output.remove(word)
    return self.vocab.get_batch(output)

  def transposing(self, word, index=None) :
    if index == None :
      output = dict()
      for i in range(len(word)-1) :
        output.update(self.transposing(word, index=i))
      return output
    if word[index] == word[index+1] :
      return {}
    w = word[:index]+word[index+1]+word[index]+word[index+2:]
    return self.vocab.get_batch([w])

  def deleting(self, word, index=None) :
    if index == None :
      output = dict()
      for i in range(len(word)) :
        output.update(self.deleting(word, index=i))
      return output
    w = word[:index]+word[index+1:]
    return self.vocab.get_batch([w])

  def get(self, word, edits={'self_exist','inserting','transposing','deleting','replacing'}, best=False) :
    output = {edit: self.edit_functions[edit](word) for edit in edits}
    if best :
      if output.get('self_exist',-1) >= 0 :
        return word
      all = dict()
      for s,t in output.items() :
        if s == 'self_exist' :
          continue
        else :
          all.update(t)
      max_count = -1
      best_word = word
      for w,c in all.items() :
        if c > max_count :
          max_count = c
          best_word = w
      return best_word
    return output

  def __getitem__(self, word) :
    return self.get(word)

  def json_object(self) :
    j = self.__dict__.copy()
    j['vocab'] = self.vocab.json_object()
    del j['edit_functions']
    return j

  @classmethod
  def loads(cls, j) :
    c = cls(Vocabulary.loads(j['vocab']))
    for attr,val in j.items() :
      if attr == 'vocab' :
        continue
      if attr == 'mcw' :
        newmcw = dict()
        for i,l in val.items() :
          newmcw[int(i)] = l
        val = newmcw
      setattr(c, attr, val)
    return c


































class Preprocessor :
  def __init__(self, source_vocab, prefix=[], suffix=[]) :
    self.normalizer = Normalizer()
    with open('./Library/NotEndCharacters.txt') as f :
      self.not_end_characters = f.read()
    with open('./Library/CityElements.json') as f :
      self.city_elements = json.load(f)
    with open('./Library/ShortNames.json') as f :
      self.short_names = json.load(f)
    with open('./Library/PersianAlphabet.txt') as f :
      self.alphabet = f.read()
    with open('./Library/SameLetters.json') as f :
      other_shapes = json.load(f)
    self.same_letters = dict()
    for s,t in other_shapes.items() :
      for c in t.split() :
        self.same_letters[c] = s
    self.vocab_process(source_vocab)
    self.SC = SpellChecker(self.vocab)
    self.SR = SpaceRemover(self.vocab)
    self.SI = SpaceInserter(self.vocab)
    self.SCI_History = dict()
    self.ESN_History = dict()
    self.prefix = prefix
    self.suffix = suffix
    with open('./Library/NumberElements.json') as f :
      self.number_elements = json.load(f)
    self.ord2norm = self.ord2norm_provider()

  def clear_history(self) :
    del self.SCI_History
    del self.ESN_History
    self.SCI_History = dict()
    self.ESN_History = dict()
    self.SI.clear_history()
    self.SR.clear_history()
    
  def ord2norm_provider(self) :
    ord2norm =      {w+'م':   w for w in self.number_elements}
    ord2norm.update({w+' م':  w for w in self.number_elements})
    ord2norm.update({w+'ام':  w for w in self.number_elements})
    ord2norm.update({w+' ام': w for w in self.number_elements})
    ord2norm.update({'اول': 'یک', 'سوم': 'سه'})
    return ord2norm

  def cleaner(self, text, save_characters='', log=False) :
    t = text
    if log : print('1/10',end='')
    t = self.normalizer.character_refinement(t)
    if log : print('\r2/10',end='')
    t = self.normalizer.normalize(t)
    if log : print('\r3/10',end='')
    t = replace_all(t, self.same_letters.items(), source_context='{}', target_context='{}')
    if log : print('\r4/10',end='')
    t = re.sub(r"[^{}\sa-z{}۰-۹٫/-]".format(self.alphabet,save_characters),r" ", t).strip()
    if log : print('\r5/10',end='')
    t = re.sub(r"(( -|)[۰-۹]+(([٫/-][۰-۹]+)+)?)",r" \1 ", t).strip()
    if log : print('\r6/10',end='')
    t = re.sub(r"([a-z]+)",r" \1 ", t.lower()).strip()
    if log : print('\r7/10',end='')
    t = re.sub(r"[٫/-]([^۰-۹])",r" \1", " "+t+" ").strip()
    t = re.sub(r"([^۰-۹ ])[٫/-]",r"\1 ", " "+t+" ").strip()
    if log : print('\r8/10',end='')
    t = re.sub(r"([٫/-])",r" \1", t).strip()
    if log : print('\r9/10',end='')
    t = re.sub(r'([\s])\1+', r'\1', t).strip()
    if log : print('\r10/10')
    return t
    
  def number_ord2norm(self, text, log=False) :
    tqd = tqdm(self.ord2norm.items()) if log else self.ord2norm.items()
    t = replace_all(' '+text+' ', tqd, target_context=' {} ')
    t = re.sub(r" ([۰-۹]+) (ا)?م ", r" \1 ", ' '+t+' ').strip()
    return t

  def number_num2word(self, text, max_trans=1000, num_tag=False, log=False) :
    try :
        from num2fawords import words as num2word
    except :
        print("module num2fawords is not installed! please install it using command `pip install num2fawords`")
        exit("module num2fawords is not installed!")
    num2w = {self.normalizer.normalize(str(i)): num2word(i) for i in range(max_trans+1)}
    tqd = tqdm(num2w.items()) if log else num2w.items()
    t = replace_all(' '+text+' ', tqd, target_context=' {} ')
    if num_tag :
      t = re.sub(r" ([۰-۹]+) ", r" <عدد> ", ' '+t+' ').strip()
    return t

  def vocab_process(self, words) :
    text = ' '.join(words)
    text = self.cleaner(text)
    words = text.split()
    self.vocab = Vocabulary()
    self.vocab.new_words(words+self.city_elements)

  def disconnect(self, text, special_characters='Z', log=False) :
    r = special_characters+r'\1'+special_characters
    t = re.sub(r'([۰-۹][۰-۹]*)', r, text).strip()
    t = re.sub(r'([a-z][a-z]*)', r, t).strip()
    tqd = self.city_elements+self.number_elements
    tqd = tqdm(list(zip(tqd,tqd)), total=len(tqd)) if log else list(zip(tqd,tqd))
    t = replace_all(' '+t+' ', tqd, target_context=' '+special_characters+'{}'+special_characters+' ')
    return t

  def update(self) :
    self.SC = SpellChecker(self.vocab)
    self.SR = SpaceRemover(self.vocab)
    self.SI = SpaceInserter(self.vocab)
    self.SCI_History = dict()

  def sr_handle(self, t, log=False) :
    SR_History = dict()
    tqd = tqdm(large_unique(t.split('Z')[::2])[0]) if log else large_unique(t.split('Z')[::2])[0]
    for part in tqd :
      part = re.sub(r'([\s])\1+', r'\1', part).strip()
      if part == '' or part == ' ' :
        continue
      SR_History.update(self.SR.find_difference(part))
    tqd = tqdm(SR_History.items()) if log else SR_History.items()
    t = replace_all(' '+t+' ', tqd)
    return t

  def sci_check(self, w) :
    if len(w) == 1 :
      return w
    wi, si = self.SI.run(w)
    wc = self.SC.get(w, best=True, edits={'inserting', 'replacing', 'deleting', 'transposing'})
    sc = self.SI.score(wc)
    s = self.SI.score((self.vocab[w]+2)**3, len(w))
    if 4*s >= si and s >= sc :
      return w
    if si > 4*sc :
      return wi
    else :
      return wc

  def sci_handle(self, t, log=False, train=False) :
    SCI_History = dict()
    tqd = tqdm(large_unique(((' '.join(t.split('Z')[::2]))).split())[0]) if log \
        else large_unique(((' '.join(t.split('Z')[::2]))).split())[0]
    for word in tqd :
      w = None
      if not train :
        w = self.SCI_History.get(word, None)
      if w == None :
        w = self.sci_check(word)
      SCI_History[word] = w
    tqd = tqdm(SCI_History.items()) if log else SCI_History.items()
    t = replace_all(' '+t+' ', tqd)
    self.SCI_History.update(SCI_History)
    return t

  def extract_short_names(self, text, vocab_is_ready=False, log=False) :
    temp_vocab = None
    if vocab_is_ready :
      temp_vocab = self.vocab
    else :
      temp_vocab = self.vocab.copy()
      temp_vocab.add_text(text)
    tqd = tqdm(self.ESN_History.items()) if log and len(self.ESN_History) > 0 else self.ESN_History.items()
    t = replace_all(' '+text+' ', tqd, target_context=' {} ')
    ts = np.array(t.split())
    candidates = dict()
    tqd = tqdm(self.short_names) if log else self.short_names
    for s in tqd :
      if len(s) > 2 :
        continue
      p = (ts==s)
      if p.sum() == 0 :
        continue
      right = np.unique(ts[:-1][p[1:]])
      for r in right :
        if r[-1] not in self.not_end_characters :
          new = r+s
          if temp_vocab[new] > temp_vocab[r] :
            candidates[r+' '+s] = new
      if s[-1] not in self.not_end_characters :
        left  = np.unique(ts[1:][p[:-1]])
        for l in left :
          new = s+l
          if temp_vocab[new] > temp_vocab[l] :
            candidates[s+' '+l] = new
    tqd = tqdm(candidates.items()) if log else candidates.items()
    t = replace_all(' '+t+' ', tqd, target_context=' {} ')
    self.ESN_History.update(candidates)
    tqd = tqdm(self.short_names.items()) if log else self.short_names.items()
    t = replace_all(' '+t+' ', tqd, target_context=' {} ')
    return t

  def analyse_prefix_suffix(self, text, log=False) :
    if log : print('1/10',end='')
    words,counts = large_unique(text.split())
    if log : print('\r2/10',end='')
    invalids = ~np.isin(words, list(self.vocab.words.keys()))
    words,counts = words[invalids],counts[invalids]
    if log : print('\r3/10',end='')
    valids = np.array(list(map(len, words))) > 3
    words,counts = words[valids],counts[valids]
    if log : print('\r4/10',end='')
    p1 = np.isin(list(map(lambda x: x[:1] , words)), self.prefix), lambda x: x[1:]
    p2 = np.isin(list(map(lambda x: x[:2] , words)), self.prefix), lambda x: x[2:]
    s1 = np.isin(list(map(lambda x: x[-1:], words)), self.suffix), lambda x: x[:-1]
    s2 = np.isin(list(map(lambda x: x[-2:], words)), self.suffix), lambda x: x[:-2]
    if log : print('\r5/10',end='')
    new_counts = dict()
    for i,(group,rule) in enumerate([p2,s2,p1,s1]) :
      if log : print('\r{}/10'.format(6+i),end='')
      for w,c in list(zip(words[group].tolist(), counts[group].tolist())) :
        neww = rule(w)
        if neww in self.vocab :
          new_counts[neww] = new_counts.get(neww,0) + c
    if log : print('\r10/10')
    self.vocab.add_words(new_counts)

  def train(self, text, new_words=False, just_look_words=True, log=True, special_characters='Z') :
    return self.run(text,
             train=True,
             new_words=new_words,
             just_look_words=just_look_words,
             log=log,
             special_characters=special_characters)

  def run(self, text, train=False, new_words=False, just_look_words=False, log=False, special_characters='Z') :
    if log : print('cleaning ...')
    t = self.cleaner(text, log=log)
    if t=='' or t.isspace():
        return ''
    if train :
      if len(self.prefix)+len(self.suffix) > 0 :
        if log : print('analysing prefix suffix ...')
        self.analyse_prefix_suffix(t, log=log)
    if log : print('extracting short names ...')
    t = self.extract_short_names(t, log=log, vocab_is_ready=not train)
    if train :
      if log : print('updating vocabulary ...')
      self.vocab.add_text(t.replace(special_characters,' '), new=new_words)
      self.vocab.add_words({ordi: self.vocab[norm] for ordi,norm in self.ord2norm.items()})
      self.update()
      if just_look_words :
        if log :
          print('='*45)
        return
    if log : print('disconnecting ...')
    t = self.disconnect(t, log=log, special_characters=special_characters)
    if log : print('space removing ...')
    t = self.sr_handle(t, log=log)
    if log : print('spell checker + space inserter are running ...')
    t = self.sci_handle(t, log=log, train=train)
    if log : print('preparing output ...')
    t = t.replace(special_characters, ' ')
    t = re.sub(r'([\s])\1+', r'\1', t).strip()
    if log :
      print('='*45)
    return t

  def batch_run(self, texts, batch=100000, split=' split ') :
    sample = large_unique(texts)[0]
    processes_sample = []
    for i in tqdm(range(sample.shape[0] // batch + 1)):
      s = self[split.join(sample[i*batch:(i+1)*batch])]
      processes_sample += s.split(split)
    return dict(zip(sample, processes_sample))

  def __getitem__(self, text) :
    return self.run(text)

  def json_object(self) :
    j = dict()
    for k,v in self.__dict__.items() :
      if k in {'normalizer','not_end_characters','city_elements','short_names','same_letters','vahed_names'} :
        continue
      if k in {'SC','SR','SI','vocab'} :
        j[k] = v.json_object()
      else :
        j[k] = v
    return j.copy()

  @classmethod
  def loads(cls, j) :
    classes = {
        'SC': SpellChecker,
        'SR': SpaceRemover,
        'SI': SpaceInserter,
        'vocab': Vocabulary
    }
    c = cls([])
    for attr,val in j.items() :
      if attr in {'SC','SR','SI','vocab'} :
        setattr(c, attr, classes[attr].loads(val))
      else :
        setattr(c, attr, val)
    return c

  def save(self, path) :
    save(self, path)

  @classmethod
  def load(cls, path) :
    return load(cls, path)