import cv2
import random
import numpy as np
import time
import Queue
import threading
import globals as g_
from concurrent.futures import ThreadPoolExecutor

W = H = 256

if g_.MODEL.lower() == 'alexnet':
    OUT_W = OUT_H = 227
elif g_.MODEL.lower() == 'vgg16':
    OUT_W = OUT_H = 224

class Image:
    def __init__(self, path, label):
        with open(path) as f:
            self.label = label
        
        self.data = self._load(path)
        self.done_mean = False
        self.normalized = False

    def _load(self, path):
        im = cv2.imread(path)
        im = cv2.resize(im, (H, W))
        # im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) #BGR!!
        assert im.shape == (H,W,3), 'BGR!'
        im = im.astype('float32')

        return im 
    
    def subtract_mean(self):
        if not self.done_mean:
            mean_bgr = (104., 116., 122.)
            for i in range(3):
                self.data[:,:,i] -= mean_bgr[i]
            
            self.done_mean = True

    def normalize(self):
        if not self.normalized:
            self.data /= 256.
            self.normalized = True
    
    def crop_center(self, size=(OUT_H, OUT_W)):
        h, w = self.data.shape[0], self.data.shape[1]
        hn, wn = size
        top = h / 2 - hn / 2
        left = w / 2 - wn / 2
        right = left + wn
        bottom = top + hn
        self.data = self.data[top:bottom, left:right, :]
    
    def crop_center(self, size=(OUT_H, OUT_W)):
        w, h = self.data.shape[0], self.data.shape[1]
        wn, hn = size
        left = w / 2 - wn / 2
        top = h / 2 - hn / 2
        self._crop(top, left, hn, wn)

    def random_crop(self, size=(227,227)):
        w, h = self.data.shape[0], self.data.shape[1]
        wn, hn = size
        left = random.randint(0, max(w - wn - 1, 0))
        top = random.randint(0, max(h - hn - 1, 0))
        self._crop(top, left, hn, wn)

    def _crop(self, top, left, h, w):
        right = left + w
        bottom = top + h
        self.data = self.data[top:bottom, left:right, :]

    def random_flip(self):
        if random.randint(0,1) == 1:
            self.data = self.data[:, ::-1, :]

class Dataset:
    def __init__(self, imagelist_file, subtract_mean, is_train, image_size=(OUT_H, OUT_W), name='dataset'):
        self.image_paths, self.labels = self._read_imagelist(imagelist_file)
        self.shuffled = False
        self.subtract_mean = subtract_mean
        self.is_train = is_train
        self.name = name
        self.image_size = image_size

        print 'image dataset "' + name + '" inited'
        print '  total size:', len(self.image_paths)


    def __getitem__(self, key):
        return self.image_paths[key], self.labels[key]

    def _read_imagelist(self, listfile):
        path_and_labels = np.loadtxt(listfile, dtype=str).tolist()
        paths, labels = zip(*[(l[0], int(l[1])) for l in path_and_labels])
        return paths, labels

    def load_image(self, path_label):
        path, label = path_label
        i = Image(path, label)       

        if not self.is_train:
            i.crop_center()
        else:
            i.random_crop()
            i.random_flip()

        if self.subtract_mean:
            i.subtract_mean()

        return i.data

    def shuffle(self):
        z = zip(self.image_paths, self.labels)
        random.shuffle(z)
        self.image_paths, self.labels = map(list, zip(*z))
        self.shuffled = True

    
    def batches(self, batch_size):
        for x,y in self._batches_fast(self.image_paths, self.labels, batch_size):
            yield x,y


    def sample_batches(self, batch_size, k):
        z = zip(self.image_paths, self.labels)
        paths, labels = map(list, zip(*random.sample(z, k)))
        for x,y in self._batches_fast(paths, labels, batch_size):
            yield x,y
    

    def _batches_fast(self, paths, labels, batch_size):
        QUEUE_END = '__QUEUE_END105834569xx' # just a random string
        n = len(paths)

        def load(inds, q):                    
            for ind in inds:
                q.put(self.load_image(paths[ind], labels[ind]))

            # indicate that I'm done
            q.put(QUEUE_END)

        def load(inds, q, batch_size):
            n = len(inds)
            with ThreadPoolExecutor(max_workers=16) as pool:
                for i in range(0, n, batch_size):
                    sub = inds[i: i + batch_size] if i < n-1 else [inds[-1]]
                    sub_paths = [paths[j] for j in sub]
                    sub_labels = [labels[j] for j in sub]
                    images = list(pool.map(self.load_image, zip(sub_paths, sub_labels)))
                    images_data = np.array(images)
                    sub_labels = np.array(sub_labels)
                    q.put((images_data, sub_labels))

            # indicate that I'm done
            q.put(None)

        q = Queue.Queue(maxsize=1024)

        # background loading images thread
        t = threading.Thread(target=load, args=(range(len(paths)), q, batch_size))
        # daemon child is killed when parent exits
        t.daemon = True
        t.start()

        h, w = self.image_size
        x = np.zeros((batch_size, h, w, 3)) 
        y = np.zeros(batch_size)

        for i in xrange(0, n, batch_size):
            starttime = time.time()
            
            item = q.get()
            if item == QUEUE_END:
                break
            
            x, y = item
            
            # print 'load batch time:', time.time()-starttime, 'sec'
            yield x, y

    def size(self):
        """ size of paths (if splitted, only count 'train', not 'val')"""
        return len(self.image_paths)


