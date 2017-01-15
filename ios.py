import os
from datetime import datetime


def get_file_list(path):
    import imghdr
    file_list = []
    name_list = []
    for (root, dirs, files) in os.walk(path):
        for file in files:
            target = os.path.join(root,file).replace("\\", "/")
            if os.path.isfile(target):
                if imghdr.what(target) != None :
                    target = target.replace(path,'')
                    file_list.append(target)
                    name_list.append(file)

    return file_list, name_list


def check_file_list(path):
    raw_path = os.path.join(path,'raw/')
    label_path = os.path.join(path, 'label/')

    raw_files, tmp_name = get_file_list(raw_path)
    label_files, tmp_name = get_file_list(label_path)

    raw_size = len(raw_files)
    label_size = len(label_files)

    if not raw_size == label_size:
        return False

    for x in xrange(raw_size):
        if not raw_files[x] == label_files[x]:
            return False

    return True


def make_output_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

    time = datetime.now()

    day_dir = time.strftime('%Y-%m-%d')
    dir_path = os.path.join(path,day_dir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    time_dir = time.strftime('%H-%M-%S')
    dir_path = os.path.join(dir_path,time_dir)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    return dir_path
