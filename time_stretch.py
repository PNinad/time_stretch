import numpy as np
import os, sys
import soundfile
from scipy.signal import windows
    
def OLA_time_stretch_audio(audio, stretch_factor, frame_size= 1024):
    """
    Implements time stretching using OLA algorithm
    """    
    FRAME_SIZE = frame_size  
    SYNTH_OVERLAP = FRAME_SIZE //2 
    STRETCH_FACTOR = stretch_factor
    SYNTHESIS_HOP_SIZE = FRAME_SIZE - SYNTH_OVERLAP
    ANALYSIS_HOP_SIZE = int(SYNTHESIS_HOP_SIZE/STRETCH_FACTOR)
    
    window = windows.hann(FRAME_SIZE)
    
    outputAudio = np.zeros(int(STRETCH_FACTOR*len(audio)))
    
    pinAnal = 0
    pinSynth = 0
    delta = SYNTH_OVERLAP

    while pinAnal < len(audio)- 2*FRAME_SIZE:
        anal_frame = audio[pinAnal:pinAnal+2*FRAME_SIZE]

        synth_frame = anal_frame[delta:delta+FRAME_SIZE]

        try:
            outputAudio[pinSynth: pinSynth+FRAME_SIZE] += window*synth_frame
        except:
            pass
        pinAnal += ANALYSIS_HOP_SIZE
        pinSynth += SYNTHESIS_HOP_SIZE

    return outputAudio


def WSOLA_time_stretch_audio(audio, stretch_factor, frame_size= 1024):
    """
    Implements time stretching using WSOLA algorithm
    """  
    
    FRAME_SIZE = frame_size  # 1024 size works well for most commony found sample rates. For samplerate of 8kHz use 512
    SYNTH_OVERLAP = FRAME_SIZE //2 
    STRETCH_FACTOR = stretch_factor
    SYNTHESIS_HOP_SIZE = FRAME_SIZE - SYNTH_OVERLAP
    ANALYSIS_HOP_SIZE = int(SYNTHESIS_HOP_SIZE/STRETCH_FACTOR)
    

    prevOverlap = np.zeros(SYNTH_OVERLAP)
    CROSS_CORR_FUNC = np.correlate
    
    window = windows.hann(FRAME_SIZE)

    outputAudio = np.zeros(int(STRETCH_FACTOR*len(audio)))
    
    pinAnal = 0
    pinSynth = 0

    while pinAnal < len(audio)- 2*FRAME_SIZE:
        anal_frame = audio[pinAnal:pinAnal+2*FRAME_SIZE]
        if not np.all(prevOverlap):
            delta = SYNTH_OVERLAP
        else:
            delta = np.argmax(CROSS_CORR_FUNC(anal_frame[:FRAME_SIZE], prevOverlap))

        synth_frame = anal_frame[delta:delta+FRAME_SIZE]

        np.copyto(prevOverlap, synth_frame[-SYNTH_OVERLAP:])

        try:
            outputAudio[pinSynth: pinSynth+FRAME_SIZE] += window*synth_frame
        except:
            pass
        pinAnal += ANALYSIS_HOP_SIZE
        pinSynth += SYNTHESIS_HOP_SIZE

    return outputAudio

def phase_unwrap(phase):
    phase = phase - 2.0 * np.pi * np.round(phase / (2.0 * np.pi))
    return phase

def PV_time_stretch_audio(audio, stretch_factor, frame_size=4096, samplerate = 44100):
    """
    Implements time stretching using phase vocoder algorithm
    """  
    FRAME_SIZE = frame_size
    STRETCH_FACTOR = stretch_factor
        
    SYNTHESIS_HOP_SIZE = FRAME_SIZE //4
    ANALYSIS_HOP_SIZE = int(SYNTHESIS_HOP_SIZE/STRETCH_FACTOR)

    FS = samplerate
    window = windows.hann(FRAME_SIZE)
    phase = np.zeros(FRAME_SIZE//2 +1)
    omega = 2*np.pi*np.arange(FRAME_SIZE//2 +1)/FRAME_SIZE
    
    outputAudio = np.zeros(int(STRETCH_FACTOR*len(audio)))

    prev_frame_fft = np.fft.rfft(np.zeros(FRAME_SIZE))
    
    del_ta = ANALYSIS_HOP_SIZE/FS
    del_ts = SYNTHESIS_HOP_SIZE/FS
    
    pinAnal = 0
    pinSynth = 0
    
    while pinAnal < len(audio) - FRAME_SIZE - ANALYSIS_HOP_SIZE:
        frame = audio[pinAnal:pinAnal+FRAME_SIZE]
        frame_fft = np.fft.rfft(frame*window)
        
        frame_fft_mag = np.abs(frame_fft)
        
        del_phi = np.angle(frame_fft) - np.angle(prev_frame_fft) - omega*del_ta
        
        del_phi = phase_unwrap(del_phi)
        
        phase = phase + del_ts * (omega + del_phi/del_ta)
        
        synth_frame = np.fft.irfft(frame_fft_mag*np.exp(1j*phase))

        try:
            outputAudio[pinSynth:pinSynth+FRAME_SIZE] += synth_frame*window
        except:
            pass
        pinAnal += ANALYSIS_HOP_SIZE
        pinSynth += SYNTHESIS_HOP_SIZE
        
    return outputAudio

def main(audio_file, stretch_factor, alg='WSOLA', frame_size= 1024):
    audio, sr = soundfile.read(audio_file)
    if len(audio.shape) == 2:
        audio = (audio[:,0] + audio[:,1]) / 2
       
    if alg=='WSOLA':
        outputAudio = WSOLA_time_stretch_audio(audio, stretch_factor, frame_size= frame_size)
    if alg=='OLA':
        outputAudio = OLA_time_stretch_audio(audio, stretch_factor, frame_size= frame_size)
    if alg=='PV':
        outputAudio = PV_time_stretch_audio(audio, stretch_factor, frame_size= frame_size, samplerate = sr)
    soundfile.write(os.path.splitext(audio_file)[0]+ f'_{stretch_factor}_{alg}.wav', outputAudio, sr)
    
if __name__ == "__main__":
    audio_file = sys.argv[1]
    stretch_factor = float(sys.argv[2])
    alg = sys.argv[3]
    if alg == 'PV':
        frame_size = 4096
    else:
        frame_size = 1024
    main(audio_file, stretch_factor, alg, frame_size)