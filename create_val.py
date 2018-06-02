import os
import datetime
import argparse
import shutil
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--val_ratio', type=int, default=20, choices=range(101))
    args = parser.parse_args()

    dataset_dir = os.path.normpath(args.dataset_dir)
    dataset_dir = os.path.abspath(dataset_dir)

    if args.output_dir:
        output_dir = args.output_dir
    else:
        now = datetime.datetime.now()
        output_dir = os.path.join(os.path.dirname(dataset_dir),
                                  "%s_%s" % (os.path.basename(dataset_dir), 
                                             now.strftime('%Y%m%d_%H%M%S'))
                                 )

    train_set = 'train'
    val_set = 'val'
    test_set = 'test'

    if os.path.exists(output_dir):
        confirm = raw_input("The existing output dir will be removed. Are you sure (yes/no): ")
        if confirm.strip() == 'yes':
            shutil.rmtree(output_dir)
        else:
            print "Stopped"
            exit()

    os.makedirs(output_dir)

    os.symlink(os.path.join(dataset_dir, test_set),
               os.path.join(output_dir, test_set))

    for dirname, subdirs, filenames in os.walk(os.path.join(dataset_dir, train_set)):
        os.makedirs(os.path.join(output_dir, os.path.relpath(dirname, dataset_dir)))

    shutil.copytree(os.path.join(output_dir, train_set),
                    os.path.join(output_dir, val_set))

    train_files = []
    val_files = []

    for dirname, subdirs, filenames in os.walk(os.path.join(dataset_dir, train_set, 'raw')):
        subset = os.path.relpath(dirname, os.path.join(dataset_dir, train_set, 'raw'))
        full_filenames = [os.path.join(subset, filename) for filename in filenames]
        np.random.shuffle(full_filenames)
        val_take = len(full_filenames) * args.val_ratio / 100
        train_files += full_filenames[val_take:]
        val_files += full_filenames[:val_take]

    for train_file in train_files:
        for sub in ['raw', 'label']:
            os.link(os.path.join(dataset_dir, train_set, sub, train_file),
                    os.path.join(output_dir, train_set, sub, train_file))

    for val_file in val_files:
        for sub in ['raw', 'label']:
            os.link(os.path.join(dataset_dir, train_set, sub, val_file),
                    os.path.join(output_dir, val_set, sub, val_file))
