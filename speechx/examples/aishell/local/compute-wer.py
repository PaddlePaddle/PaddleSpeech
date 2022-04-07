#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import re, sys, unicodedata
import codecs

remove_tag = True
spacelist= [' ', '\t', '\r', '\n']
puncts = ['!', ',', '?',
          '、', '。', '！', '，', '；', '？',
          '：', '「', '」', '︰',  '『', '』', '《', '》']

def characterize(string) :
  res = []
  i = 0
  while i < len(string):
    char = string[i]
    if char in puncts:
      i += 1
      continue
    cat1 = unicodedata.category(char)
    #https://unicodebook.readthedocs.io/unicode.html#unicode-categories
    if cat1 == 'Zs' or cat1 == 'Cn' or char in spacelist: # space or not assigned
       i += 1
       continue
    if cat1 == 'Lo': # letter-other
       res.append(char)
       i += 1
    else:
       # some input looks like: <unk><noise>, we want to separate it to two words.
       sep = ' '
       if char == '<': sep = '>'
       j = i+1
       while j < len(string):
         c = string[j]
         if ord(c) >= 128 or (c in spacelist) or (c==sep):
           break
         j += 1
       if j < len(string) and string[j] == '>':
         j += 1
       res.append(string[i:j])
       i = j
  return res

def stripoff_tags(x):
  if not x: return ''
  chars = []
  i = 0; T=len(x)
  while i < T:
    if x[i] == '<':
      while i < T and x[i] != '>':
         i += 1
      i += 1
    else:
      chars.append(x[i])
      i += 1
  return ''.join(chars)


def normalize(sentence, ignore_words, cs, split=None):
    """ sentence, ignore_words are both in unicode
    """
    new_sentence = []
    for token in sentence:
        x = token
        if not cs:
           x = x.upper()
        if x in ignore_words:
           continue
        if remove_tag:
          x = stripoff_tags(x)
        if not x:
          continue
        if split and x in split:
          new_sentence += split[x]
        else:
          new_sentence.append(x)
    return new_sentence

class Calculator :
  def __init__(self) :
    self.data = {}
    self.space = []
    self.cost = {}
    self.cost['cor'] = 0
    self.cost['sub'] = 1
    self.cost['del'] = 1
    self.cost['ins'] = 1
  def calculate(self, lab, rec) :
    # Initialization
    lab.insert(0, '')
    rec.insert(0, '')
    while len(self.space) < len(lab) :
      self.space.append([])
    for row in self.space :
      for element in row :
        element['dist'] = 0
        element['error'] = 'non'
      while len(row) < len(rec) :
        row.append({'dist' : 0, 'error' : 'non'})
    for i in range(len(lab)) :
      self.space[i][0]['dist'] = i
      self.space[i][0]['error'] = 'del'
    for j in range(len(rec)) :
      self.space[0][j]['dist'] = j
      self.space[0][j]['error'] = 'ins'
    self.space[0][0]['error'] = 'non'
    for token in lab :
      if token not in self.data and len(token) > 0 :
        self.data[token] = {'all' : 0, 'cor' : 0, 'sub' : 0, 'ins' : 0, 'del' : 0}
    for token in rec :
      if token not in self.data and len(token) > 0 :
        self.data[token] = {'all' : 0, 'cor' : 0, 'sub' : 0, 'ins' : 0, 'del' : 0}
    # Computing edit distance
    for i, lab_token in enumerate(lab) :
      for j, rec_token in enumerate(rec) :
        if i == 0 or j == 0 :
          continue
        min_dist = sys.maxsize
        min_error = 'none'
        dist = self.space[i-1][j]['dist'] + self.cost['del']
        error = 'del'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        dist = self.space[i][j-1]['dist'] + self.cost['ins']
        error = 'ins'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        if lab_token == rec_token :
          dist = self.space[i-1][j-1]['dist'] + self.cost['cor']
          error = 'cor'
        else :
          dist = self.space[i-1][j-1]['dist'] + self.cost['sub']
          error = 'sub'
        if dist < min_dist :
          min_dist = dist
          min_error = error
        self.space[i][j]['dist'] = min_dist
        self.space[i][j]['error'] = min_error
    # Tracing back
    result = {'lab':[], 'rec':[], 'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    i = len(lab) - 1
    j = len(rec) - 1
    while True :
      if self.space[i][j]['error'] == 'cor' : # correct
        if len(lab[i]) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['cor'] = self.data[lab[i]]['cor'] + 1
          result['all'] = result['all'] + 1
          result['cor'] = result['cor'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, rec[j])
        i = i - 1
        j = j - 1
      elif self.space[i][j]['error'] == 'sub' : # substitution
        if len(lab[i]) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['sub'] = self.data[lab[i]]['sub'] + 1
          result['all'] = result['all'] + 1
          result['sub'] = result['sub'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, rec[j])
        i = i - 1
        j = j - 1
      elif self.space[i][j]['error'] == 'del' : # deletion
        if len(lab[i]) > 0 :
          self.data[lab[i]]['all'] = self.data[lab[i]]['all'] + 1
          self.data[lab[i]]['del'] = self.data[lab[i]]['del'] + 1
          result['all'] = result['all'] + 1
          result['del'] = result['del'] + 1
        result['lab'].insert(0, lab[i])
        result['rec'].insert(0, "")
        i = i - 1
      elif self.space[i][j]['error'] == 'ins' : # insertion
        if len(rec[j]) > 0 :
          self.data[rec[j]]['ins'] = self.data[rec[j]]['ins'] + 1
          result['ins'] = result['ins'] + 1
        result['lab'].insert(0, "")
        result['rec'].insert(0, rec[j])
        j = j - 1
      elif self.space[i][j]['error'] == 'non' : # starting point
        break
      else : # shouldn't reach here
        print('this should not happen , i = {i} , j = {j} , error = {error}'.format(i = i, j = j, error = self.space[i][j]['error']))
    return result
  def overall(self) :
    result = {'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    for token in self.data :
      result['all'] = result['all'] + self.data[token]['all']
      result['cor'] = result['cor'] + self.data[token]['cor']
      result['sub'] = result['sub'] + self.data[token]['sub']
      result['ins'] = result['ins'] + self.data[token]['ins']
      result['del'] = result['del'] + self.data[token]['del']
    return result
  def cluster(self, data) :
    result = {'all':0, 'cor':0, 'sub':0, 'ins':0, 'del':0}
    for token in data :
      if token in self.data :
        result['all'] = result['all'] + self.data[token]['all']
        result['cor'] = result['cor'] + self.data[token]['cor']
        result['sub'] = result['sub'] + self.data[token]['sub']
        result['ins'] = result['ins'] + self.data[token]['ins']
        result['del'] = result['del'] + self.data[token]['del']
    return result
  def keys(self) :
      return list(self.data.keys())

def width(string):
  return sum(1 + (unicodedata.east_asian_width(c) in "AFW") for c in string)

def default_cluster(word) :
  unicode_names = [ unicodedata.name(char) for char in word ]
  for i in reversed(range(len(unicode_names))) :
    if unicode_names[i].startswith('DIGIT') :  # 1
      unicode_names[i] = 'Number'  # 'DIGIT'
    elif (unicode_names[i].startswith('CJK UNIFIED IDEOGRAPH') or
          unicode_names[i].startswith('CJK COMPATIBILITY IDEOGRAPH')) :
      # 明 / 郎
      unicode_names[i] = 'Mandarin'  # 'CJK IDEOGRAPH'
    elif (unicode_names[i].startswith('LATIN CAPITAL LETTER') or
          unicode_names[i].startswith('LATIN SMALL LETTER')) :
      # A / a
      unicode_names[i] = 'English'  # 'LATIN LETTER'
    elif unicode_names[i].startswith('HIRAGANA LETTER') :  # は こ め
      unicode_names[i] = 'Japanese'  # 'GANA LETTER'
    elif (unicode_names[i].startswith('AMPERSAND') or
          unicode_names[i].startswith('APOSTROPHE') or
          unicode_names[i].startswith('COMMERCIAL AT') or
          unicode_names[i].startswith('DEGREE CELSIUS') or
          unicode_names[i].startswith('EQUALS SIGN') or
          unicode_names[i].startswith('FULL STOP') or
          unicode_names[i].startswith('HYPHEN-MINUS') or
          unicode_names[i].startswith('LOW LINE') or
          unicode_names[i].startswith('NUMBER SIGN') or
          unicode_names[i].startswith('PLUS SIGN') or
          unicode_names[i].startswith('SEMICOLON')) :
      # & / ' / @ / ℃ / = / . / - / _ / # / + / ;
      del unicode_names[i]
    else :
      return 'Other'
  if len(unicode_names) == 0 :
      return 'Other'
  if len(unicode_names) == 1 :
      return unicode_names[0]
  for i in range(len(unicode_names)-1) :
    if unicode_names[i] != unicode_names[i+1] :
      return 'Other'
  return unicode_names[0]

def usage() :
  print("compute-wer.py : compute word error rate (WER) and align recognition results and references.")
  print("         usage : python compute-wer.py [--cs={0,1}] [--cluster=foo] [--ig=ignore_file] [--char={0,1}] [--v={0,1}] [--padding-symbol={space,underline}] test.ref test.hyp > test.wer")

if __name__ == '__main__':
  if len(sys.argv) == 1 :
    usage()
    sys.exit(0)
  calculator = Calculator()
  cluster_file = ''
  ignore_words = set()
  tochar = False
  verbose= 1
  padding_symbol= ' '
  case_sensitive = False
  max_words_per_line = sys.maxsize
  split = None
  while len(sys.argv) > 3:
     a = '--maxw='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):]
        del sys.argv[1]
        max_words_per_line = int(b)
        continue
     a = '--rt='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        remove_tag = (b == 'true') or (b != '0')
        continue
     a = '--cs='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        case_sensitive = (b == 'true') or (b != '0')
        continue
     a = '--cluster='
     if sys.argv[1].startswith(a):
       cluster_file = sys.argv[1][len(a):]
       del sys.argv[1]
       continue
     a = '--splitfile='
     if sys.argv[1].startswith(a):
       split_file = sys.argv[1][len(a):]
       del sys.argv[1]
       split = dict()
       with codecs.open(split_file, 'r', 'utf-8') as fh:
         for line in fh:  # line in unicode
           words = line.strip().split()
           if len(words) >= 2:
             split[words[0]] = words[1:]
       continue
     a = '--ig='
     if sys.argv[1].startswith(a):
       ignore_file = sys.argv[1][len(a):]
       del sys.argv[1]
       with codecs.open(ignore_file, 'r', 'utf-8') as fh:
         for line in fh:  # line in unicode
           line = line.strip()
           if len(line) > 0:
             ignore_words.add(line)
       continue
     a = '--char='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        tochar = (b == 'true') or (b != '0')
        continue
     a = '--v='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        verbose=0
        try:
          verbose=int(b)
        except:
           if b == 'true' or b != '0':
              verbose = 1
        continue
     a = '--padding-symbol='
     if sys.argv[1].startswith(a):
        b = sys.argv[1][len(a):].lower()
        del sys.argv[1]
        if b == 'space':
          padding_symbol= ' '
        elif b == 'underline':
          padding_symbol= '_'
        continue
     if True or sys.argv[1].startswith('-'):
        #ignore invalid switch
        del sys.argv[1]
        continue

  if not case_sensitive:
     ig=set([w.upper() for w in ignore_words])
     ignore_words = ig

  default_clusters = {}
  default_words = {}

  ref_file = sys.argv[1]
  hyp_file = sys.argv[2]
  rec_set = {}
  if split and not case_sensitive:
     newsplit = dict()
     for w in split:
        words = split[w]
        for i in range(len(words)):
           words[i] = words[i].upper()
        newsplit[w.upper()] = words
     split = newsplit

  with codecs.open(hyp_file, 'r', 'utf-8') as fh:
     for line in fh:
        if tochar:
            array = characterize(line)
        else:
            array = line.strip().split()
        if len(array)==0: continue
        fid = array[0]
        rec_set[fid] = normalize(array[1:], ignore_words, case_sensitive, split)

  # compute error rate on the interaction of reference file and hyp file
  for line in open(ref_file, 'r', encoding='utf-8') :
    if tochar:
          array = characterize(line)
    else:
          array = line.rstrip('\n').split()
    if len(array)==0: continue
    fid = array[0]
    if fid not in rec_set:
       continue
    lab = normalize(array[1:], ignore_words, case_sensitive, split)
    rec = rec_set[fid]
    if verbose:
      print('\nutt: %s' % fid)

    for word in rec + lab :
      if word not in default_words :
         default_cluster_name = default_cluster(word)
         if default_cluster_name not in default_clusters :
           default_clusters[default_cluster_name] = {}
         if word not in default_clusters[default_cluster_name] :
           default_clusters[default_cluster_name][word] = 1
         default_words[word] = default_cluster_name

    result = calculator.calculate(lab, rec)
    if verbose:
      if result['all'] != 0 :
        wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
      else :
        wer = 0.0
      print('WER: %4.2f %%' % wer, end = ' ')
      print('N=%d C=%d S=%d D=%d I=%d' %
          (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
      space = {}
      space['lab'] = []
      space['rec'] = []
      for idx in range(len(result['lab'])) :
        len_lab = width(result['lab'][idx])
        len_rec = width(result['rec'][idx])
        length = max(len_lab, len_rec)
        space['lab'].append(length-len_lab)
        space['rec'].append(length-len_rec)
      upper_lab = len(result['lab'])
      upper_rec = len(result['rec'])
      lab1, rec1 = 0, 0
      while lab1 < upper_lab or rec1 < upper_rec:
         if verbose > 1:
             print('lab(%s):' % fid.encode('utf-8'), end = ' ')
         else:
             print('lab:', end = ' ')
         lab2 = min(upper_lab, lab1 + max_words_per_line)
         for idx in range(lab1, lab2):
           token = result['lab'][idx]
           print('{token}'.format(token = token), end = '')
           for n in range(space['lab'][idx]) :
             print(padding_symbol, end = '')
           print(' ',end='')
         print()
         if verbose > 1:
            print('rec(%s):' % fid.encode('utf-8'), end = ' ')
         else:
            print('rec:', end = ' ')
         rec2 = min(upper_rec, rec1 + max_words_per_line)
         for idx in range(rec1, rec2):
           token = result['rec'][idx]
           print('{token}'.format(token = token), end = '')
           for n in range(space['rec'][idx]) :
             print(padding_symbol, end = '')
           print(' ',end='')
         print('\n', end='\n')
         lab1 = lab2
         rec1 = rec2

  if verbose:
    print('===========================================================================')
    print()

  result = calculator.overall()
  if result['all'] != 0 :
    wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
  else :
    wer = 0.0
  print('Overall -> %4.2f %%' % wer, end = ' ')
  print('N=%d C=%d S=%d D=%d I=%d' %
        (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
  if not verbose:
     print()

  if verbose:
   for cluster_id in default_clusters :
     result = calculator.cluster([ k for k in default_clusters[cluster_id] ])
     if result['all'] != 0 :
        wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
     else :
        wer = 0.0
     print('%s -> %4.2f %%' % (cluster_id, wer), end = ' ')
     print('N=%d C=%d S=%d D=%d I=%d' %
          (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
   if len(cluster_file) > 0 : # compute separated WERs for word clusters
     cluster_id = ''
     cluster = []
     for line in open(cluster_file, 'r', encoding='utf-8') :
       for token in line.decode('utf-8').rstrip('\n').split() :
        # end of cluster reached, like </Keyword>
        if token[0:2] == '</' and token[len(token)-1] == '>' and \
           token.lstrip('</').rstrip('>') == cluster_id :
          result = calculator.cluster(cluster)
          if result['all'] != 0 :
            wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
          else :
            wer = 0.0
          print('%s -> %4.2f %%' % (cluster_id, wer), end = ' ')
          print('N=%d C=%d S=%d D=%d I=%d' %
                (result['all'], result['cor'], result['sub'], result['del'], result['ins']))
          cluster_id = ''
          cluster = []
        # begin of cluster reached, like <Keyword>
        elif token[0] == '<' and token[len(token)-1] == '>' and \
             cluster_id == '' :
          cluster_id = token.lstrip('<').rstrip('>')
          cluster = []
        # general terms, like WEATHER / CAR / ...
        else :
          cluster.append(token)
   print()
   print('===========================================================================')
