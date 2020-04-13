import sys
import json
import os.path
import re

# ToDo Implement clean cache function ? 
# ToDo implement parse functions for the dataset


# returns True if the given root path contains a valid vocalset installation
def is_vocalset_root(path):
    test_file = 'female1/arpeggios/belt/f1_arpeggios_belt_c_a'

    if(os.path.isfile(os.path.join(path, test_file + '.ogg'))):
        return True
        
    if(os.path.isfile(os.path.join(path, test_file + '.wav'))):
        return True

    return False

# returns the location of the vocalset.json configuration file an containing folder
def get_config_json_path():
    build_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, 'build'))
    json_path = os.path.join(build_dir, 'vocalset.json')

    return build_dir, json_path 

# function returns the root path of the VocalSet installation
def cache_root_path(path):
    build_dir, json_file = get_config_json_path()

    data = {}
    data['vocalset'] = []
    data['vocalset'].append({'root_path': path})

    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    with open(json_file, 'w+') as outfile:
        json.dump(data, outfile)
        print('Updated VocalSet root path: \n' + path)
        return

    raise Exception('Could not store the VocalSet root path in ' + json_file)
    

# function returns the cached vocalset root path
def get_root_path(check_path = True):
    _, json_file_path = get_config_json_path()

    with open(json_file_path) as json_file:
        data = json.load(json_file)
        for cfg in data['vocalset']:
            root_path = cfg['root_path']
            if(check_path and not is_vocalset_root(root_path)):
                raise Exception('Cached VocalSet root path seems to be invalid: ' + root_path)

            return root_path


    
    raise Exception('Could not find the json containing a cached VocalSet root path. Try calling dataset.py <root_path>')




# updates the vocalset root path in the configuration vocalset.json. Returns true if path is valid and cach could be updated
def set_vocalset_root_path(dataset_root_path):

    # check if it's a valid folder
    if(not os.path.isdir(dataset_root_path)):
        return False

    dataset_root_path = os.path.abspath(dataset_root_path)

    # test path
    if(is_vocalset_root(dataset_root_path)):
        cache_root_path(dataset_root_path)
        return True

    # test path/FULL 
    dataset_root_path = os.path.join(dataset_root_path, 'FULL')
    if(is_vocalset_root(dataset_root_path)):
        cache_root_path(dataset_root_path)
        return True
        
    return False

def get_sample(types, modes, vowels, singers):

    # helper finds folders that fit a pattern describes as a list of filters
    def resolve_folders(path, filter_list):

        # return this path as valid if we have no filters
        if(len(filter_list) == 0):
            
            return [path]

        # take our filter and truncate the list
        dir_filters = filter_list[0]
        filter_list = filter_list[1:]

        directories = []
        for sub_dir in os.listdir(path):
            sub_path = os.path.join(path, sub_dir)
            if os.path.isdir(sub_path):
                if (sub_dir in dir_filters) or ('*' in dir_filters) or (len(dir_filters) == 0):        
                    found = resolve_folders(os.path.join(path, sub_path), filter_list)
                    directories.extend(found)

        return directories


    # get root path
    root_path = get_root_path()

    # cleaning input 
    listify = lambda arg: [arg] if (type(arg) == str) else arg
    vowels  = listify(vowels)
    modes   = listify(modes)
    types   = listify(types)
    singers = listify(singers)

    # replace short version 'f1' - 'm1' with 'female1' - 'male1'
    for i, singer in enumerate(singers):
        singers[i] = re.sub(r"^(m)(.*)", r'male\2', singers[i])
        singers[i] = re.sub(r"^(f)(.*)", r'female\2', singers[i])

    
    # collect valid folders
    paths = resolve_folders(root_path, [singers, types, modes])

    # collect valid files within folders
    files = []
    for path in paths:
        for file in os.listdir(path):
            full_file = os.path.join(path,file)
            if os.path.isfile(full_file) and (file.endswith(".wav") or file.endswith(".ogg")):                
                for vowel in vowels:
                    if(re.match(f".*_{vowel}\.wav", file)):
                        files.append(full_file)


    return files


# calling daaset.py <root_path> sets / caches the VocalSet root path in build/vocalset.json
if __name__ == '__main__':
    
    if(len(sys.argv) == 2 and sys.argv[1] == '--get'):
        print(get_root_path())
        sys.exit(0)
    
    if(len(sys.argv) == 3 and sys.argv[1] == '--set'):
        dataset_root_path = os.path.normpath(sys.argv[2])
        was_set = set_vocalset_root_path(dataset_root_path)
            
        if(not was_set):
            print('Could not find a valid VocalSet root at ' + dataset_root_path)            
            sys.exit(-1)

        sys.exit(0)



    print('dataset.py: Expected dataset.py --set <path> or dataset.py --get')
    
    sys.exit(-1)
    
    

