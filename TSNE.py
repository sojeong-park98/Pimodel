# scikit-learn과 matplotlib 설치
import numpy as np
import torch
from ConvLarge import ConvLarge
from dataset import TransSVHN
from torchvision.transforms import transforms as T
from torch.utils.data.dataloader import DataLoader

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# sckiptlearn 라이브러리에 있는 TSNE 모델 사용하겠습니다.
# n_component는 차원 수
tsne = TSNE(n_components=2)

# 랜덤시드 셋팅
# SVHN 데이터를 불러와서 embedding space 출력해보겠습니다.
root ='./SVHN'
n_sample = 10000 # 샘플 몇개 쓸지
n_batch = 256 # 데이터 불러올 때 배치 수

if __name__ == '__main__':

    #-----------EMBEDDING 출력에 사용할 모델 불러오기------------#
    # 모델 정의
    model = ConvLarge(num_classes=10)
    # 훈련된 모델 불러오기
    checkpoint = torch.load('best_model.pt')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # -----------EMBEDDING 출력에 사용할 데이터 불러오기------------#
    # simple transformation
    init_trans = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010))])
    # 테스트 데이터
    test_set = TransSVHN(
        root=root,
        training=False,
        transform=init_trans,
        target_transform=None,
        supervised=False)
    test_loader = DataLoader(test_set,
                             batch_size=n_batch,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False,
                             pin_memory=True)

    # 임베딩 스페이스 출력을 위해 특징과, 레이블 값을 저장해둘 배열
    feat_list = None
    label_list = None
    with torch.no_grad():
        len_samples = 0
        for _, (input, label) in enumerate(test_loader):
            # 모델을 돌면서 특징과 라벨을 저장합니다.
            # ConvLarge.py의 model의 forward에 return_feature option을 활성화 합니다.
            feat = model(input, return_feature=True)[1]
            # 텐서 형태로 특징과 레이블 벡터를 저장해줍니다.
            if feat_list == None:
                feat_list = feat
                label_list = label
            else:
                feat_list = torch.cat([feat_list, feat], 0)
                label_list = torch.cat([label_list, label], 0)
                len_samples += len(input)

            # n_samples가 넘으면 멈춰줍니다.
            if len_samples >= n_sample:
                break
    # tsne로 넘기기 위해 텐서에서 넘파이 배열로 변환해줍니다.
    feat_list = np.array(feat_list)
    label_list = np.array(label_list)

    # tsne 모델에 fitting 시켜줍니다.
    embedding = tsne.fit_transform(feat_list)

    # 출력된 embedding을 확인합니다.
    plt.scatter(embedding[:,0], embedding[:,1], c=label_list)
    plt.show()
