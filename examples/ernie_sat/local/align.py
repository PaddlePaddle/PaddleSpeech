""" Usage:
    align.py wavfile trsfile outwordfile outphonefile
"""
import os
import sys

PHONEME = 'tools/aligner/english_envir/english2phoneme/phoneme'
MODEL_DIR_EN = 'tools/aligner/english'
MODEL_DIR_ZH = 'tools/aligner/mandarin'
HVITE = 'tools/htk/HTKTools/HVite'
HCOPY = 'tools/htk/HTKTools/HCopy'


def get_unk_phns(word_str: str):
    tmpbase = '/tmp/tp.'
    f = open(tmpbase + 'temp.words', 'w')
    f.write(word_str)
    f.close()
    os.system(PHONEME + ' ' + tmpbase + 'temp.words' + ' ' + tmpbase +
              'temp.phons')
    f = open(tmpbase + 'temp.phons', 'r')
    lines2 = f.readline().strip().split()
    f.close()
    phns = []
    for phn in lines2:
        phons = phn.replace('\n', '').replace(' ', '')
        seq = []
        j = 0
        while (j < len(phons)):
            if (phons[j] > 'Z'):
                if (phons[j] == 'j'):
                    seq.append('JH')
                elif (phons[j] == 'h'):
                    seq.append('HH')
                else:
                    seq.append(phons[j].upper())
                j += 1
            else:
                p = phons[j:j + 2]
                if (p == 'WH'):
                    seq.append('W')
                elif (p in ['TH', 'SH', 'HH', 'DH', 'CH', 'ZH', 'NG']):
                    seq.append(p)
                elif (p == 'AX'):
                    seq.append('AH0')
                else:
                    seq.append(p + '1')
                j += 2
        phns.extend(seq)
    return phns


def words2phns(line: str):
    '''
    Args:
        line (str): input text.
        eg: for that reason cover is impossible to be given.
    Returns:
        List[str]: phones of input text.
            eg:
            ['F', 'AO1', 'R', 'DH', 'AE1', 'T', 'R', 'IY1', 'Z', 'AH0', 'N', 'K', 'AH1', 'V', 'ER0',
            'IH1', 'Z', 'IH2', 'M', 'P', 'AA1', 'S', 'AH0', 'B', 'AH0', 'L', 'T', 'UW1', 'B', 'IY1', 
            'G', 'IH1', 'V', 'AH0', 'N']

        Dict(str, str): key - idx_word
                        value - phones
            eg:
            {'0_FOR': ['F', 'AO1', 'R'], '1_THAT': ['DH', 'AE1', 'T'], '2_REASON': ['R', 'IY1', 'Z', 'AH0', 'N'],
            '3_COVER': ['K', 'AH1', 'V', 'ER0'], '4_IS': ['IH1', 'Z'], '5_IMPOSSIBLE': ['IH2', 'M', 'P', 'AA1', 'S', 'AH0', 'B', 'AH0', 'L'],
            '6_TO': ['T', 'UW1'], '7_BE': ['B', 'IY1'], '8_GIVEN': ['G', 'IH1', 'V', 'AH0', 'N']}
    '''
    dictfile = MODEL_DIR_EN + '/dict'
    line = line.strip()
    words = []
    for pun in [',', '.', ':', ';', '!', '?', '"', '(', ')', '--', '---']:
        line = line.replace(pun, ' ')
    for wrd in line.split():
        if (wrd[-1] == '-'):
            wrd = wrd[:-1]
        if (wrd[0] == "'"):
            wrd = wrd[1:]
        if wrd:
            words.append(wrd)
    ds = set([])
    word2phns_dict = {}
    with open(dictfile, 'r') as fid:
        for line in fid:
            word = line.split()[0]
            ds.add(word)
            if word not in word2phns_dict.keys():
                word2phns_dict[word] = " ".join(line.split()[1:])

    phns = []
    wrd2phns = {}
    for index, wrd in enumerate(words):
        if wrd == '[MASK]':
            wrd2phns[str(index) + "_" + wrd] = [wrd]
            phns.append(wrd)
        elif (wrd.upper() not in ds):
            wrd2phns[str(index) + "_" + wrd.upper()] = get_unk_phns(wrd)
            phns.extend(get_unk_phns(wrd))
        else:
            wrd2phns[str(index) +
                     "_" + wrd.upper()] = word2phns_dict[wrd.upper()].split()
            phns.extend(word2phns_dict[wrd.upper()].split())
    return phns, wrd2phns


def words2phns_zh(line: str):
    dictfile = MODEL_DIR_ZH + '/dict'
    line = line.strip()
    words = []
    for pun in [
            ',', '.', ':', ';', '!', '?', '"', '(', ')', '--', '---', u'，',
            u'。', u'：', u'；', u'！', u'？', u'（', u'）'
    ]:
        line = line.replace(pun, ' ')
    for wrd in line.split():
        if (wrd[-1] == '-'):
            wrd = wrd[:-1]
        if (wrd[0] == "'"):
            wrd = wrd[1:]
        if wrd:
            words.append(wrd)

    ds = set([])
    word2phns_dict = {}
    with open(dictfile, 'r') as fid:
        for line in fid:
            word = line.split()[0]
            ds.add(word)
            if word not in word2phns_dict.keys():
                word2phns_dict[word] = " ".join(line.split()[1:])

    phns = []
    wrd2phns = {}
    for index, wrd in enumerate(words):
        if wrd == '[MASK]':
            wrd2phns[str(index) + "_" + wrd] = [wrd]
            phns.append(wrd)
        elif (wrd.upper() not in ds):
            print("出现非法词错误,请输入正确的文本...")
        else:
            wrd2phns[str(index) + "_" + wrd] = word2phns_dict[wrd].split()
            phns.extend(word2phns_dict[wrd].split())

    return phns, wrd2phns


def prep_txt_zh(line: str, tmpbase: str, dictfile: str):

    words = []
    line = line.strip()
    for pun in [
            ',', '.', ':', ';', '!', '?', '"', '(', ')', '--', '---', u'，',
            u'。', u'：', u'；', u'！', u'？', u'（', u'）'
    ]:
        line = line.replace(pun, ' ')
    for wrd in line.split():
        if (wrd[-1] == '-'):
            wrd = wrd[:-1]
        if (wrd[0] == "'"):
            wrd = wrd[1:]
        if wrd:
            words.append(wrd)

    ds = set([])
    with open(dictfile, 'r') as fid:
        for line in fid:
            ds.add(line.split()[0])

    unk_words = set([])
    with open(tmpbase + '.txt', 'w') as fwid:
        for wrd in words:
            if (wrd not in ds):
                unk_words.add(wrd)
            fwid.write(wrd + ' ')
        fwid.write('\n')
    return unk_words


def prep_txt_en(line: str, tmpbase, dictfile):

    words = []

    line = line.strip()
    for pun in [',', '.', ':', ';', '!', '?', '"', '(', ')', '--', '---']:
        line = line.replace(pun, ' ')
    for wrd in line.split():
        if (wrd[-1] == '-'):
            wrd = wrd[:-1]
        if (wrd[0] == "'"):
            wrd = wrd[1:]
        if wrd:
            words.append(wrd)

    ds = set([])
    with open(dictfile, 'r') as fid:
        for line in fid:
            ds.add(line.split()[0])

    unk_words = set([])
    with open(tmpbase + '.txt', 'w') as fwid:
        for wrd in words:
            if (wrd.upper() not in ds):
                unk_words.add(wrd.upper())
            fwid.write(wrd + ' ')
        fwid.write('\n')

    #generate pronounciations for unknows words using 'letter to sound'
    with open(tmpbase + '_unk.words', 'w') as fwid:
        for unk in unk_words:
            fwid.write(unk + '\n')
    try:
        os.system(PHONEME + ' ' + tmpbase + '_unk.words' + ' ' + tmpbase +
                  '_unk.phons')
    except Exception:
        print('english2phoneme error!')
        sys.exit(1)

    #add unknown words to the standard dictionary, generate a tmp dictionary for alignment 
    fw = open(tmpbase + '.dict', 'w')
    with open(dictfile, 'r') as fid:
        for line in fid:
            fw.write(line)
    f = open(tmpbase + '_unk.words', 'r')
    lines1 = f.readlines()
    f.close()
    f = open(tmpbase + '_unk.phons', 'r')
    lines2 = f.readlines()
    f.close()
    for i in range(len(lines1)):
        wrd = lines1[i].replace('\n', '')
        phons = lines2[i].replace('\n', '').replace(' ', '')
        seq = []
        j = 0
        while (j < len(phons)):
            if (phons[j] > 'Z'):
                if (phons[j] == 'j'):
                    seq.append('JH')
                elif (phons[j] == 'h'):
                    seq.append('HH')
                else:
                    seq.append(phons[j].upper())
                j += 1
            else:
                p = phons[j:j + 2]
                if (p == 'WH'):
                    seq.append('W')
                elif (p in ['TH', 'SH', 'HH', 'DH', 'CH', 'ZH', 'NG']):
                    seq.append(p)
                elif (p == 'AX'):
                    seq.append('AH0')
                else:
                    seq.append(p + '1')
                j += 2

        fw.write(wrd + ' ')
        for s in seq:
            fw.write(' ' + s)
        fw.write('\n')
    fw.close()


def prep_mlf(txt: str, tmpbase: str):

    with open(tmpbase + '.mlf', 'w') as fwid:
        fwid.write('#!MLF!#\n')
        fwid.write('"' + tmpbase + '.lab"\n')
        fwid.write('sp\n')
        wrds = txt.split()
        for wrd in wrds:
            fwid.write(wrd.upper() + '\n')
            fwid.write('sp\n')
        fwid.write('.\n')


def _get_user():
    return os.path.expanduser('~').split("/")[-1]


def alignment(wav_path: str, text: str):
    '''
    intervals: List[phn, start, end]
    '''
    tmpbase = '/tmp/' + _get_user() + '_' + str(os.getpid())

    #prepare wav and trs files
    try:
        os.system('sox ' + wav_path + ' -r 16000 ' + tmpbase + '.wav remix -')
    except Exception:
        print('sox error!')
        return None

    #prepare clean_transcript file
    try:
        prep_txt_en(line=text, tmpbase=tmpbase, dictfile=MODEL_DIR_EN + '/dict')
    except Exception:
        print('prep_txt error!')
        return None

    #prepare mlf file
    try:
        with open(tmpbase + '.txt', 'r') as fid:
            txt = fid.readline()
        prep_mlf(txt, tmpbase)
    except Exception:
        print('prep_mlf error!')
        return None

    #prepare scp
    try:
        os.system(HCOPY + ' -C ' + MODEL_DIR_EN + '/16000/config ' + tmpbase +
                  '.wav' + ' ' + tmpbase + '.plp')
    except Exception:
        print('HCopy error!')
        return None

    #run alignment
    try:
        os.system(HVITE + ' -a -m -t 10000.0 10000.0 100000.0 -I ' + tmpbase +
                  '.mlf -H ' + MODEL_DIR_EN + '/16000/macros -H ' + MODEL_DIR_EN
                  + '/16000/hmmdefs -i ' + tmpbase + '.aligned ' + tmpbase +
                  '.dict ' + MODEL_DIR_EN + '/monophones ' + tmpbase +
                  '.plp 2>&1 > /dev/null')
    except Exception:
        print('HVite error!')
        return None

    with open(tmpbase + '.txt', 'r') as fid:
        words = fid.readline().strip().split()
    words = txt.strip().split()
    words.reverse()

    with open(tmpbase + '.aligned', 'r') as fid:
        lines = fid.readlines()
    i = 2
    intervals = []
    word2phns = {}
    current_word = ''
    index = 0
    while (i < len(lines)):
        splited_line = lines[i].strip().split()
        if (len(splited_line) >= 4) and (splited_line[0] != splited_line[1]):
            phn = splited_line[2]
            pst = (int(splited_line[0]) / 1000 + 125) / 10000
            pen = (int(splited_line[1]) / 1000 + 125) / 10000
            intervals.append([phn, pst, pen])
            # splited_line[-1]!='sp'
            if len(splited_line) == 5:
                current_word = str(index) + '_' + splited_line[-1]
                word2phns[current_word] = phn
                index += 1
            elif len(splited_line) == 4:
                word2phns[current_word] += ' ' + phn
        i += 1
    return intervals, word2phns


def alignment_zh(wav_path: str, text: str):
    tmpbase = '/tmp/' + _get_user() + '_' + str(os.getpid())

    #prepare wav and trs files
    try:
        os.system('sox ' + wav_path + ' -r 16000 -b 16 ' + tmpbase +
                  '.wav remix -')

    except Exception:
        print('sox error!')
        return None

    #prepare clean_transcript file
    try:
        unk_words = prep_txt_zh(
            line=text, tmpbase=tmpbase, dictfile=MODEL_DIR_ZH + '/dict')
        if unk_words:
            print('Error! Please add the following words to dictionary:')
            for unk in unk_words:
                print("非法words: ", unk)
    except Exception:
        print('prep_txt error!')
        return None

    #prepare mlf file
    try:
        with open(tmpbase + '.txt', 'r') as fid:
            txt = fid.readline()
        prep_mlf(txt, tmpbase)
    except Exception:
        print('prep_mlf error!')
        return None

    #prepare scp
    try:
        os.system(HCOPY + ' -C ' + MODEL_DIR_ZH + '/16000/config ' + tmpbase +
                  '.wav' + ' ' + tmpbase + '.plp')
    except Exception:
        print('HCopy error!')
        return None

    #run alignment
    try:
        os.system(HVITE + ' -a -m -t 10000.0 10000.0 100000.0 -I ' + tmpbase +
                  '.mlf -H ' + MODEL_DIR_ZH + '/16000/macros -H ' + MODEL_DIR_ZH
                  + '/16000/hmmdefs -i ' + tmpbase + '.aligned ' + MODEL_DIR_ZH
                  + '/dict ' + MODEL_DIR_ZH + '/monophones ' + tmpbase +
                  '.plp 2>&1 > /dev/null')

    except Exception:
        print('HVite error!')
        return None

    with open(tmpbase + '.txt', 'r') as fid:
        words = fid.readline().strip().split()
    words = txt.strip().split()
    words.reverse()

    with open(tmpbase + '.aligned', 'r') as fid:
        lines = fid.readlines()

    i = 2
    intervals = []
    word2phns = {}
    current_word = ''
    index = 0
    while (i < len(lines)):
        splited_line = lines[i].strip().split()
        if (len(splited_line) >= 4) and (splited_line[0] != splited_line[1]):
            phn = splited_line[2]
            pst = (int(splited_line[0]) / 1000 + 125) / 10000
            pen = (int(splited_line[1]) / 1000 + 125) / 10000
            intervals.append([phn, pst, pen])
            # splited_line[-1]!='sp'
            if len(splited_line) == 5:
                current_word = str(index) + '_' + splited_line[-1]
                word2phns[current_word] = phn
                index += 1
            elif len(splited_line) == 4:
                word2phns[current_word] += ' ' + phn
        i += 1
    return intervals, word2phns
