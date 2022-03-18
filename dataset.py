import os
import random
import torchvision
import numpy as np
from PIL import Image


def get_labeled_files(targets, samples, keep_file='./txt/split_4000.txt', training=True):
    new_targets, new_samples = [], []
    # 훈련셋이고, labeled 데이터를 불러옴 (텍스트 파일에 미리 정의된 데이터 4000개)
    if training and (keep_file is not None):
        assert os.path.exists(keep_file), 'keep file does not exist'
        with open(keep_file, 'r') as rfile:
            for line in rfile:
                indx = int(line.split('\n')[0])
                new_targets.append(targets[indx])
                new_samples.append(samples[indx])
    return np.array(new_targets), np.array(new_samples)


class TransSVHN(torchvision.datasets.SVHN):
    def __init__(
        self,
        root,
        seed=123,
        keep_file=None,
        num_labeled=400,
        training=True,
        transform=None,
        target_transform=None,
        supervised=True,
    ):
        super().__init__(root, 'train', None, None, True)
        self.supervised = supervised
        self.transform = transform
        self.training = training

        if keep_file is not None:
            random.seed(seed)
            src = []
            num_classes = 10
            shot = int(num_labeled / num_classes)
            for val in range(0,10):
                cur_src = [index for index, value in enumerate(self.labels) if value == val]
                idxes = random.sample(cur_src, shot)
                for idx in idxes:
                    src.append(idx)

            with open(keep_file, "w") as f:
                for i in range(len(src)):
                    f.write(str(src[i]) + '\n')

        # Labeled 데이터를 불러와야 하는 경우 텍스트 파일에 접근
        if self.supervised:
            self.labels, self.data = get_labeled_files(self.labels, self.data, keep_file=keep_file, training=training)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(img.transpose(1,2,0))

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            # Labeled 데이터나 test 데이터는 한가지 버전의 augmented 데이터만 출력
            if self.supervised or not self.training:
                return self.transform(img), target
            # Unlabeled 데이터에 대한 훈련시 동일한 이미지에 대한 두가지 버전의 augmented 데이터 출력
            else:
                img_1 = self.transform(img)
                img_2 = self.transform(img)
                return img_1, img_2, target
        return img, target
