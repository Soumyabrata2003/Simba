import numpy as np
import pickle
from torch.utils.data import Dataset

from feeders import tools


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        """
        :param data_path:
        :param label_path:
        :param split: training set or test set
        :param random_choose: If true, randomly choose a portion of the input sequence
        :param random_shift: If true, randomly pad zeros at the begining or end of sequence
        :param random_move:
        :param random_rot: rotate skeleton around xyz axis
        :param window_size: The length of the output sequence
        :param normalization: If true, normalize input sequence
        :param debug: If true, only use the first 100 samples
        :param use_mmap: If true, use mmap mode to load data, which can save the running memory
        :param bone: use bone modality or not
        :param vel: use motion modality or not
        :param only_label: only load label for ensemble score compute
        """

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        self.parse_data()
        if normalization:
            self.get_mean_map()

    # def load_data(self):
    #     # data: N C V T M
    #     npz_data = np.load(self.data_path)
    #     if self.split == 'train':
    #         self.data = npz_data['x_train']
    #         self.label = np.where(npz_data['y_train'] > 0)[1]
    #         self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
    #     elif self.split == 'test':
    #         self.data = npz_data['x_test']
    #         self.label = np.where(npz_data['y_test'] > 0)[1]
    #         self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
    #     else:
    #         raise NotImplementedError('data split only supports train/test')
    #     N, T, _ = self.data.shape
    #     self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def load_data(self):
        # data: N C V T M
        try:
            with open(self.label_path) as f:
                self.sample_name, self.label = pickle.load(f)
        except:
            # for pickle file from python2
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f, encoding='latin1')

        # load data
        if self.use_mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
            # print(self.data.shape,len(self.label))
        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

    def parse_data(self):
        # N, C, T, V, M = self.data.shape
        # clean_data=[]
        # clean_label=[]
        # for i in range(N):
        #     sample=self.data[i]
        #     if (sample.shape[1]>0):
        #         clean_data.append(sample)
        #         clean_label.append(self.label[i])
        # self.data=np.stack(clean_data)
        # self.label=clean_label
        # print(self.data.shape,len(self.label))
        N, C, T, V, M = self.data.shape
        clean_data=[]
        clean_label=[]
        for i in range(N):
            sample=self.data[i]
            valid_frame_num = np.sum(sample.sum(0).sum(-1).sum(-1) != 0)
            if (valid_frame_num>5):
                clean_data.append(sample)
                clean_label.append(self.label[i])
        self.data=np.stack(clean_data)
        self.label=clean_label
        print(self.data.shape,len(self.label))
        
    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)
        if self.bone:
            from .bone_pairs_kinetics import kinetics_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in kinetics_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]

            # keep spine center's trajectory !!! modified on July 4th, 2022
            bone_data_numpy[:, :, 20] = data_numpy[:, :, 20]
            data_numpy = bone_data_numpy

        ## for joint modality
        ## separate trajectory from relative coordinate to each frame's spine center
        # else:
            # # there's a freedom to choose the direction of local coordinate axes!
            # trajectory = data_numpy[:, :, 20]
            # trajectory = data_numpy[:, :, 1]
            # let spine of each frame be the joint coordinate center
            # data_numpy = data_numpy - data_numpy[:, :, 20:21]
            # data_numpy = data_numpy - data_numpy[:, :, 1:2]
            # ## works well with bone, but has negative effect with joint and distance gate
            # data_numpy[:, :, 20] = trajectory
            # data_numpy[:, :, 1] = trajectory
            # if self.normalization:
            #     data_numpy = (data_numpy - self.mean_map) / self.std_map
            # if self.repeat_pad:
            #     data_numpy = tools.repeat_pading(data_numpy)
            # if self.random_shift:
            #     data_numpy = tools.random_shift(data_numpy)
            # if self.random_choose:
            #     data_numpy = tools.random_choose(data_numpy, self.window_size)
            # elif self.window_size > 0:
            #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
            # if self.random_move:
            #     data_numpy = tools.random_move(data_numpy)



        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]

            data_numpy[:, -1] = 0

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
