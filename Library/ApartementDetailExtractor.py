from Library.Utils import replace_all
from num2fawords import words as num2word
import re
import json

class ADE : # Appartement Detail Extractor
  def __init__(self) :
    with open('./Library/NumberElements.json') as f :
      number_elements = json.load(f)
    ord2norm =      {w+'م':   w for w in number_elements}
    ord2norm.update({w+' م':  w for w in number_elements})
    ord2norm.update({w+'ام':  w for w in number_elements})
    ord2norm.update({w+' ام': w for w in number_elements})
    ord2norm.update({'اول': 'یک', 'سوم': 'سه'})
    self.ord2norm = ord2norm
    with open('./Library/UnitAlphabetNames.json') as f :
      self.unit_transform = json.load(f)

  def plateno(self, t) :
    if 'پلاک ' not in t :
      return None
    x = re.findall(r'(پلاک ?-? ([۰-۹]+) جدید)', t)
    if len(x) > 0 :
      return int(x[0][1])
    x = re.findall(r'(پلاک ?-? ([۰-۹]+) قدیم ?و? ?-? ([۰-۹]+) جدید)', t)
    if len(x) > 0 :
      return int(x[0][2])
    x = re.findall(r'(پلاک جدید ([۰-۹]+))', t)
    if len(x) > 0 :
      return int(x[0][1])
    x = re.findall(r'(پلاک ?-? ([۰-۹]+))', t)
    if len(x) > 0 :
      return int(x[0][1])
    x = re.findall(r'(پلاک (ه ای|های) ([۰-۹]+))', t)
    if len(x) > 0 :
      return int(x[0][2])
    x = replace_all(' '+t+' ', self.ord2norm.items())
    for i in range(100) :
        if 'پلاک '+num2word(i) in x :
          return i
    return None

  def floorno(self, t) :
    if 'همکف' in t :
      return 0
    x = re.findall(r'(طبقه ?ی? (-? ?[۰-۹]+))', t)
    if len(x) > 0 :
      return int(x[0][1].replace(' ',''))
    x = re.findall(r'(زیرزمین -? ?([۰-۹]+))', t)
    if len(x) > 0 :
      return int(-x[0][1])
    tt = replace_all(' '+t+' ', self.ord2norm.items())
    for i in range(100) :
      n = num2word(i)
      if 'طبقه '+n in tt or 'طبقه ی '+n in tt :
        return i
      if 'زیرزمین '+n in tt :
        return -i
    if 'زیرزمین' in t :
      return -1
    if 'زیر زمین' in t :
      return -1
    if 'طبقه اخر' in t :
      return 0.5
    if 'طبقه' in tt :
      x = tt.split('طبقه')[-1]
      x = x.split('واحد')[0]
      for w in x :
        if w.isdigit() :
          return int(w)
      for w in x :
        for i in range(100) :
          if num2word(i) == w :
            return i
    return None

  def unit(self, t) :
    if 'واحد ' in t :
      x = re.findall(r'(واحد (شماره|قطعه|طبقه)? ?ی? ?(منفی)? ?(-? ?[۰-۹]+))', t)
      if len(x) > 0 :
        return int(x[0][-1].replace(' ','')) if 'منفی' not in x[0] else -int(x[0][-1])
      x = re.findall(r'(واحد (.*)?(شمالی|جنوبی|شرقی|غربی|میانی|جنوب غربی|جنوب شرقی|شمال غربی|شمال شرقی|میانی غربی|میانی شرقی|میانی شمالی|میانی جنوبی|مرکزی|همکف))', t)
      if len(x) > 0 :
        return x[0][-1]
      x = re.findall(r'(واحد (.*)?(جنوب غرب|جنوب شرق|شمال غرب|شمال شرق|میان شرقی|میان غربی|میان شمالی|میان جنوبی|میان شرق|میان غرب|میان شمال|میان جنوب))', t)
      if len(x) > 0 :
        return x[0][-1]+'ی'
      x = re.findall(r'(واحد (.*)?(شمال|جنوب|شرق|غرب|میان|مرکز))', t)
      if len(x) > 0 :
        return x[0][-1]+'ی'
      x = re.findall(r'(واحد -? ?({}|[a-z]) ?([۱-۹]+)?)'.format('|'.join(list(self.unit_transform.keys()))), t)
      if len(x) > 0 :
        x = x[0][-2]
        x = self.unit_transform.get(x, x)
        if x[0][-1] != '' :
          x+= '-'+x[0][-1]
        return x
      x = re.findall(r'(واحد -? ?سمت (\S+))', t)
      if len(x) > 0 :
        return 'سمت '+x[0][-1]
      tt = replace_all(' '+t+' ', self.ord2norm.items())
      for i in range(100) :
        n = num2word(i)
        x = re.findall(r'(واحد (شماره|قطعه|طبقه)? ?ی? ?(منفی)? ?({}))'.format(n), tt)
        if len(x) > 0 :
          return i if 'منفی' not in x[0] else -i
    
    else :
      x = re.findall(r'(طبقه (.*)?(شمالی|جنوبی|شرقی|غربی|میانی|جنوب غربی|جنوب شرقی|شمال غربی|شمال شرقی|میانی غربی|میانی شرقی|میانی شمالی|میانی جنوبی|مرکزی|همکف))', t)
      if len(x) > 0 :
        return x[0][-1]
      x = re.findall(r'(طبقه (.*)?(جنوب غرب|جنوب شرق|شمال غرب|شمال شرق|میان شرقی|میان غربی|میان شمالی|میان جنوبی|میان شرق|میان غرب|میان شمال|میان جنوب))', t)
      if len(x) > 0 :
        return x[0][-1]+'ی'
      x = re.findall(r'(طبقه (.*)?(شمال|جنوب|شرق|غرب|میان|مرکز))', t)
      if len(x) > 0 :
        return x[0][-1]+'ی'

    return None

  def run(self, t) :
    return {
        "plateno": self.plateno(t),
        "floorno": self.floorno(t),
        "unit": self.unit(t)
    }

  def __getitem__(self, t) :
    return self.run(t)