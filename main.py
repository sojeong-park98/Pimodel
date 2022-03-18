import math
import torch
from itertools import cycle
from ConvLarge import ConvLarge
from dataset import TransSVHN
from torchvision.transforms import transforms as T
from torch.utils.data.dataloader import DataLoader
from utils import AddGaussianNoise
import torch.backends.cudnn as cudnn

# 쿠다 사용이 가능한지 확인
if torch.cuda.is_available():
    #print("Use CUDA")
    cuda = True
    cudnn.benchmark = True
else:
    print("Use unavailable")
    cuda = False

# 랜덤시드 셋팅
torch.manual_seed(1234)

# 하이퍼 파라미터 셋팅
lr = 0.001
image_size = 28  # 입력 이미지를 28 X 28로 리사이즈
root = './SVHN' # 데이터셋 위치
l_batch = 100 # 레이블 데이터 배치사이즈
u_batch = 100 # 언레이블 데이터 배치사이즈
t_batch = 256 # 테스트 배치 사이즈

ramp_up = 30 # 80에퐄까지 ssl 웨이트가 가우시안 ramp-up 형태로 상승
total_epoch = 100 # 전체 에퐄 수

if __name__ == '__main__':
    # augmentation with random gaussian
    trans = T.Compose([
        T.RandomCrop(size=image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)),
        AddGaussianNoise(p=0.5, mean=0., std=0.15)])
    # simple transformation
    init_trans = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010))])

    # 랜덤 augmentation을 한 4000장의 labeled 데이터
    supervised_set = TransSVHN(
        root=root,
        keep_file=f'./txt/SVHN_400.txt',
        num_labeled=400,
        training=True,
        transform=trans,
        target_transform=None,
        supervised=True)
    # 랜덤 augmentation을 한 전체 데이터 (unlabeled)
    unsupervised_set = TransSVHN(
        root=root,
        training=True,
        transform=trans,
        target_transform=None,
        supervised=False)
    # 테스트 데이터
    test_set = TransSVHN(
        root=root,
        training=False,
        transform=init_trans,
        target_transform=None,
        supervised=False)

    sampler = None
    l_loader = DataLoader(supervised_set,
                          sampler=sampler,
                          batch_size=l_batch,
                          shuffle=True,
                          num_workers=0,
                          drop_last=True,
                          pin_memory=True)

    u_loader = DataLoader(unsupervised_set,
                          batch_size=u_batch,
                          shuffle=True,
                          num_workers=0,
                          drop_last=True,
                          pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=t_batch,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False,
                             pin_memory=True)

    # 모델 정의
    model = ConvLarge(num_classes=10)
    if cuda:
        model = model.cuda()
    # 아담 옵티마이저 사용
    optim = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999))

    # 손실함수 정의
    cls_fn = torch.nn.CrossEntropyLoss()
    mse_fn = torch.nn.MSELoss()

    start_epoch = 0
    best_acc = -1

    # Training
    for epoch in range(start_epoch, total_epoch):
        model.train()
        for idx, ((l_image, l_label), (u_image, p_image, u_label)) in enumerate(zip(cycle(l_loader), u_loader)):
            if cuda:
                l_image, l_label = l_image.cuda(), l_label.long().cuda()
                u_image, p_image, u_label = u_image.cuda(), p_image.cuda(), u_label.long().cuda()

            optim.zero_grad()

            #모델 출력 얻기
            l_z = model(l_image)
            u_z = model(u_image)
            p_z = model(p_image)
            l_z, u_z, p_z = l_z.float(), u_z.float(), p_z.float()
            l_preds, u_preds, p_preds = l_z.argmax(dim=1), u_z.argmax(dim=1), p_z.argmax(dim=1)
            # Labeled 데이터의 출력에 대한 cross entropy 계산
            cls_loss = cls_fn(l_z, l_label.detach())
            # Unlabled 데이터와 purturbated 데이터의 출력에 대한 MSE loss 계산
            mse_loss = mse_fn(p_z, u_z.detach())
            # mse loss에 대한 가중치 계산
            coef = math.exp(-5 * (1 - min(epoch / ramp_up, 1)) ** 2)
            loss = cls_loss + coef * mse_loss

            loss.backward()
            optim.step()

            l_acc = l_preds.eq(l_label).float().mean()
            u_acc = u_preds.eq(u_label).float().mean()

            if idx % 30 == 0:
                lr = optim.param_groups[0]["lr"]
                log = 'EPOCH:[%03d/%03d], iter:[%04d/%04d], l_loss: %.03f, u_loss: %.03f, l_acc: %.03f, u_acc: %.03f'
                print(log % (epoch + 1, total_epoch, idx + 1, len(u_loader), cls_loss, mse_loss, l_acc, u_acc))

        # 테스트 데이터로 정확도 계산
        model.eval()
        with torch.no_grad():
            t_acc = []
            for _, (input, label) in enumerate(test_loader):
                if cuda:
                    input, label = input.cuda(), label.long().cuda()
                z = model(input)
                z.float()
                preds = z.argmax(dim=1)
                t_acc.append(preds.eq(label).float().mean())
            test_acc = sum(t_acc) / len(t_acc)

        if test_acc >= best_acc:
            best_acc=test_acc
            save_dict = {'model': model.state_dict(),
                         'opt': optim.state_dict(),
                         'epoch': epoch + 1}
            torch.save(save_dict, './best_model.pt')
        print('The test accuracy is %.03f' % test_acc)
        print('The best accuracy is %.03f' % best_acc)
