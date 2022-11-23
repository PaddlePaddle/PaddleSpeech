

forked from `<https://github.com/jameslyons/python_speech_features>`_

check the readme therein for the usages

It has been modified to produce the same results as with the compute-mfcc-feats and compute-fbank-feats (check their default parameters first) commands in Kaldi.
 
-------------------------------

The compute-mfcc-feats pipeline:

src/featbin/Compute-mfcc-feats.cc
    
    Mfcc mfcc(mfcc_opts)  --> src/feat/Feature-mfcc.h
    
                                 struct MfccOptions
                                 
                                 typedef OfflineFeatureTpl<MfccComputer> Mfcc --> src/feat/Feature-common.h
           
                                 MfccComputer()  --> src/feat/Feature-mfcc.cc
                                 
                                                         ComputeDctMatrix()  --> src/matrix/Matrix-functions.cc
                                                         
                                                         ComputeLifterCoeffs()  --> src/feat/Mel-computations.cc
  
    
    for each utterance:
    mfcc.ComputeFeatures()

src/feat/Feature-common-inl.h

    OfflineFeatureTpl<F>::ComputeFeatures()
    
        Compute()
        
            ExtractWindow()  --> src/feat/Feature-window.cc
                                     
                                     ProcessWindow()
                                         
                                         Dither, remove_dc_offset, log_energy_pre_window, Preemphasize, window
            
            computer_.Compute() --> src/feat/Feature-mfcc.cc
               
                                      MfccComputer::Compute()
                                      
                                          const MelBanks &mel_banks --> Mel-computations.cc
                                          
                                          srfft_
                                        
                                          ComputerPowerSpectrum()
                                          
                                          mel_banks.Compute()
                                          
                                          mel_energies_.ApplyLog()
                                          
                                          dct, cepstral_lifter
                                          
