
import os;

def getDirectoryList(path):
    directoryList = []

    #return nothing if path is a file
    if os.path.isfile(path):
        return []

    #add dir to directorylist if it contains .txt files
    if len([f for f in os.listdir(path) if f.endswith('.wav')])>0:
        directoryList.append(path)

    for d in os.listdir(path):
        new_path = os.path.join(path, d)
        if os.path.isdir(new_path):
            directoryList += getDirectoryList(new_path)

    return directoryList

src = '../VocalSetWAV'

wavFolders = getDirectoryList(src)

for folder in wavFolders:
    print(folder)
    
    os.system('python -m crepe -s 5 ' + folder)

