def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


test_dict=unpickle("/home/gong/tf_learning/cifar-10-python/cifar-10-batches-py/test_batch")
print(test_dict['labels'])