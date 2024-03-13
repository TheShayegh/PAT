class PAT :
  def __init__(self, preprocessor, nb3, ade, accurate_restorer, urban_hierarchy) :
    self.p = preprocessor
    self.nb3 = nb3
    self.ade = ade
    self.ar = accurate_restorer
    self.urban_hierarchy = urban_hierarchy

  def postcode_finder(self, prediction, details) :
    if details['plateno'] == None :
      return [],"Invalid Plateno!"
    plateno_postcodes = self.urban_hierarchy[prediction['parish']][prediction['avenue']].get(str(details['plateno']), None)
    if type(plateno_postcodes) == type(None) :
      return [],"Unmatchable Plateno!"
    if details['floorno'] == None :
      return plateno_postcodes,"Invalid Floorno!"
    floorno_postcodes = plateno_postcodes.get(str(details['floorno']), None)
    if type(floorno_postcodes) == type(None) :
      return plateno_postcodes,"Unmatchable Floorno!"
    return floorno_postcodes,"Valid Appartement!"

  def run(self, address) :
    clean = self.p[address]
    prediction,_ = self.nb3[clean]
    details = self.ade[clean]
    prediction =  self.ar.apply_label_cond(prediction, clean, 'avenue')
    prediction = self.ar.apply_plateno_cond(prediction, details['plateno'], keep_layers=['avenue'])
    prediction = self.ar.apply_plateno_cond(prediction, details['plateno'], keep_layers=['parish'])
    postcodes,comment = self.postcode_finder(prediction[0],details)
    return {
        'PAT-understandable address': clean,
        'appartement info': details,
        'most probable suggestion': prediction[0],
        'suggestions': prediction,
        'possible postcodes': postcodes,
        'comment': comment
    }

  def __getitem__(self, address) :
    return self.run(address)