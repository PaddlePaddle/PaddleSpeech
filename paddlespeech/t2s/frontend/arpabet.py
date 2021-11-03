# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from paddlespeech.t2s.frontend.phonectic import Phonetics
"""
A phonology system with ARPABET symbols and limited punctuations. The G2P 
conversion is done by g2p_en.

Note that g2p_en does not handle words with hypen well. So make sure the input
sentence is first normalized.
"""
from paddlespeech.t2s.frontend.vocab import Vocab
from g2p_en import G2p


class ARPABET(Phonetics):
    """A phonology for English that uses ARPABET as the phoneme vocabulary.
    See http://www.speech.cs.cmu.edu/cgi-bin/cmudict for more details.
    Phoneme Example Translation
        ------- ------- -----------
        AA	odd     AA D
        AE	at	AE T
        AH	hut	HH AH T
        AO	ought	AO T
        AW	cow	K AW
        AY	hide	HH AY D
        B 	be	B IY
        CH	cheese	CH IY Z
        D 	dee	D IY
        DH	thee	DH IY
        EH	Ed	EH D
        ER	hurt	HH ER T
        EY	ate	EY T
        F 	fee	F IY
        G 	green	G R IY N
        HH	he	HH IY
        IH	it	IH T
        IY	eat	IY T
        JH	gee	JH IY
        K 	key	K IY
        L 	lee	L IY
        M 	me	M IY
        N 	knee	N IY
        NG	ping	P IH NG
        OW	oat	OW T
        OY	toy	T OY
        P 	pee	P IY
        R 	read	R IY D
        S 	sea	S IY
        SH	she	SH IY
        T 	tea	T IY
        TH	theta	TH EY T AH
        UH	hood	HH UH D
        UW	two	T UW
        V 	vee	V IY
        W 	we	W IY
        Y 	yield	Y IY L D
        Z 	zee	Z IY
        ZH	seizure	S IY ZH ER
    """
    phonemes = [
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER',
        'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW',
        'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UW', 'UH', 'V', 'W', 'Y', 'Z',
        'ZH'
    ]
    punctuations = [',', '.', '?', '!']
    symbols = phonemes + punctuations
    _stress_to_no_stress_ = {
        'AA0': 'AA',
        'AA1': 'AA',
        'AA2': 'AA',
        'AE0': 'AE',
        'AE1': 'AE',
        'AE2': 'AE',
        'AH0': 'AH',
        'AH1': 'AH',
        'AH2': 'AH',
        'AO0': 'AO',
        'AO1': 'AO',
        'AO2': 'AO',
        'AW0': 'AW',
        'AW1': 'AW',
        'AW2': 'AW',
        'AY0': 'AY',
        'AY1': 'AY',
        'AY2': 'AY',
        'EH0': 'EH',
        'EH1': 'EH',
        'EH2': 'EH',
        'ER0': 'ER',
        'ER1': 'ER',
        'ER2': 'ER',
        'EY0': 'EY',
        'EY1': 'EY',
        'EY2': 'EY',
        'IH0': 'IH',
        'IH1': 'IH',
        'IH2': 'IH',
        'IY0': 'IY',
        'IY1': 'IY',
        'IY2': 'IY',
        'OW0': 'OW',
        'OW1': 'OW',
        'OW2': 'OW',
        'OY0': 'OY',
        'OY1': 'OY',
        'OY2': 'OY',
        'UH0': 'UH',
        'UH1': 'UH',
        'UH2': 'UH',
        'UW0': 'UW',
        'UW1': 'UW',
        'UW2': 'UW'
    }

    def __init__(self):
        self.backend = G2p()
        self.vocab = Vocab(self.phonemes + self.punctuations)

    def _remove_vowels(self, phone):
        return self._stress_to_no_stress_.get(phone, phone)

    def phoneticize(self, sentence, add_start_end=False):
        """ Normalize the input text sequence and convert it into pronunciation sequence.
    
        Parameters
        -----------
        sentence: str
            The input text sequence.
    
        Returns
        ----------
        List[str]
            The list of pronunciation sequence.
        """
        phonemes = [
            self._remove_vowels(item) for item in self.backend(sentence)
        ]
        if add_start_end:
            start = self.vocab.start_symbol
            end = self.vocab.end_symbol
            phonemes = [start] + phonemes + [end]
        phonemes = [item for item in phonemes if item in self.vocab.stoi]
        return phonemes

    def numericalize(self, phonemes):
        """ Convert pronunciation sequence into pronunciation id sequence.
        
        Parameters
        -----------
        phonemes: List[str]
            The list of pronunciation sequence.
    
        Returns
        ----------
        List[int]
            The list of pronunciation id sequence.
        """
        ids = [self.vocab.lookup(item) for item in phonemes]
        return ids

    def reverse(self, ids):
        """ Reverse the list of pronunciation id sequence to a list of pronunciation sequence.
        
        Parameters
        -----------
        ids: List[int]
            The list of pronunciation id sequence.
    
        Returns
        ----------
        List[str]
            The list of pronunciation sequence.
        """
        return [self.vocab.reverse(i) for i in ids]

    def __call__(self, sentence, add_start_end=False):
        """ Convert the input text sequence into pronunciation id sequence.
    
        Parameters
        -----------
        sentence: str
            The input text sequence.
    
        Returns
        ----------
        List[str]
            The list of pronunciation id sequence.
        """
        return self.numericalize(
            self.phoneticize(sentence, add_start_end=add_start_end))

    @property
    def vocab_size(self):
        """ Vocab size.
        """
        # 47 = 39 phones + 4 punctuations + 4 special tokens
        return len(self.vocab)


class ARPABETWithStress(Phonetics):
    phonemes = [
        'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
        'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D',
        'DH', 'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2',
        'F', 'G', 'HH', 'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K',
        'L', 'M', 'N', 'NG', 'OW0', 'OW1', 'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R',
        'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW0', 'UW1', 'UW2', 'V',
        'W', 'Y', 'Z', 'ZH'
    ]
    punctuations = [',', '.', '?', '!']
    symbols = phonemes + punctuations

    def __init__(self):
        self.backend = G2p()
        self.vocab = Vocab(self.phonemes + self.punctuations)

    def phoneticize(self, sentence, add_start_end=False):
        """ Normalize the input text sequence and convert it into pronunciation sequence.
    
        Parameters
        -----------
        sentence: str
            The input text sequence.
    
        Returns
        ----------
        List[str]
            The list of pronunciation sequence.
        """
        phonemes = self.backend(sentence)
        if add_start_end:
            start = self.vocab.start_symbol
            end = self.vocab.end_symbol
            phonemes = [start] + phonemes + [end]
        phonemes = [item for item in phonemes if item in self.vocab.stoi]
        return phonemes

    def numericalize(self, phonemes):
        """ Convert pronunciation sequence into pronunciation id sequence.
        
        Parameters
        -----------
        phonemes: List[str]
            The list of pronunciation sequence.
    
        Returns
        ----------
        List[int]
            The list of pronunciation id sequence.
        """
        ids = [self.vocab.lookup(item) for item in phonemes]
        return ids

    def reverse(self, ids):
        """ Reverse the list of pronunciation id sequence to a list of pronunciation sequence.
        
        Parameters
        -----------
        ids: List[int]
            The list of pronunciation id sequence.
    
        Returns
        ----------
        List[str]
            The list of pronunciation sequence.
        """
        return [self.vocab.reverse(i) for i in ids]

    def __call__(self, sentence, add_start_end=False):
        """ Convert the input text sequence into pronunciation id sequence.
    
        Parameters
        -----------
        sentence: str
            The input text sequence.
    
        Returns
        ----------
        List[str]
            The list of pronunciation id sequence.
        """
        return self.numericalize(
            self.phoneticize(sentence, add_start_end=add_start_end))

    @property
    def vocab_size(self):
        """ Vocab size.
        """
        # 77 = 69 phones + 4 punctuations + 4 special tokens
        return len(self.vocab)
