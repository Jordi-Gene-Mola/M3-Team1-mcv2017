import platform
import cPickle

train_images_filenames = cPickle.load(open('../dataset/train_images_filenames.dat', 'r'))
test_images_filenames = cPickle.load(open('../dataset/test_images_filenames.dat', 'r'))
path_to_replace = '../../'
target = '../'
save_paths = False
if platform.system() == 'Windows':
    #FIXME: Replace '/' with '\' correctly, escaping char not working...
    train_images_filenames = map(lambda x: x.replace('/', "\\"), train_images_filenames)
    test_images_filenames = map(lambda x: x.replace('/', "\\"), test_images_filenames)
    #train_images_filenames = map(lambda x: str(backslash).join(x.split('/')), train_images_filenames)
    #test_images_filenames = map(lambda x: str(backslash).join(x.split('/')), test_images_filenames)

    print('Refactored filenames:')
    print(str(train_images_filenames))
    print(str(test_images_filenames))

train_images_filenames = map(lambda x: x.replace(path_to_replace, target), train_images_filenames)
test_images_filenames = map(lambda x: x.replace(path_to_replace, target), test_images_filenames)
print(str(train_images_filenames))
print(str(test_images_filenames))
if save_paths:
    cPickle.dump(train_images_filenames, open('./train_images_filenames_map.dat', 'w'))
    cPickle.dump(test_images_filenames, open('./test_images_filenames_map.dat', 'w'))
