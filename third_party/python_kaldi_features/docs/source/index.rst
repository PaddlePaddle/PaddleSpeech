.. python_speech_features documentation master file, created by
   sphinx-quickstart on Thu Oct 31 16:49:58 2013.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to python_speech_features's documentation!
==================================================

This library provides common speech features for ASR including MFCCs and filterbank energies.
If you are not sure what MFCCs are, and would like to know more have a look at this MFCC tutorial: 
http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/.

You will need numpy and scipy to run these files. The code for this project is available at https://github.com/jameslyons/python_speech_features .

Supported features:

- :py:meth:`python_speech_features.mfcc` - Mel Frequency Cepstral Coefficients
- :py:meth:`python_speech_features.fbank` - Filterbank Energies
- :py:meth:`python_speech_features.logfbank` - Log Filterbank Energies
- :py:meth:`python_speech_features.ssc` - Spectral Subband Centroids

To use MFCC features::

    from python_speech_features import mfcc
    from python_speech_features import logfbank
    import scipy.io.wavfile as wav
    
    (rate,sig) = wav.read("file.wav")
    mfcc_feat = mfcc(sig,rate)
    fbank_feat = logfbank(sig,rate)
    
    print(fbank_feat[1:3,:])

From here you can write the features to a file etc.

Functions provided in python_speech_features module
-------------------------------------
   
.. automodule:: python_speech_features.base
    :members:
    

Functions provided in sigproc module
------------------------------------
.. automodule:: python_speech_features.sigproc
    :members:


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

