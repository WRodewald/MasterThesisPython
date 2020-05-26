import soundfile 

def read( src_file ):

    data, fs = soundfile.read(src_file)
    return data, fs

