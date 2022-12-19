import torch
import json
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet
from vit_pytorch.efficient import ViT
from linformer import Linformer
from torch.utils.data import DataLoader


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    batch_size = 32

    # torchvision自带的图片预处理
    # transforms.Resize()将图片调整为指定大小
    # transforms.RandomHorizontalFlip()图片随机水平翻转
    # ......参考torchvision官网教学

    data_transform = {
        "train": transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

        # 所有图片转化为相同大,把图片随机翻转，把图片数据集转换为Pytorch张量,用数据集的均值和标准差把数据集归一化

        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ])}

    # torchvision中的datasets加载训练集
    train_dataset = datasets.ImageFolder(root="E:/project helmet/train/", transform=data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 查看类别
    # print(train_dataset.classes)
    # 查看类别名，及对应的标签(根据分的文件夹的名字的顺序来确定的类别)
    # print(train_dataset.class_to_idx)
    # 查看路径里所有的图片，及对应的标签(返回从所有文件夹中得到的图片的路径以及其类别)
    # print(train_dataset.imgs)
    # 查看迭代器里的内容，返回一个列表
    # print(next(iter(train_loader)))
    # 显示第一张图片转换成的tensor，tuple里一个张量shape为tensor[3,244,244]和一个标量表示类别
    # print(train_dataset[0])
    # tensor[3, 244, 244]
    # print(train_dataset[0][0].shape)
    # 显示第一张图片转换成的tensor所对应的类
    # print(train_dataset[0][1])

    # # helmet_list = train_dataset.class_to_idx
    # # # 将字典进行编码，最终生成class_indices.json文件
    # # cla_dict = dict((val, key) for key, val in helmet_list.items())
    # # json_str = json.dumps(cla_dict, indent=4)
    # # with open('class_indices.json', 'w') as json_file:
    # #     json_file.write(json_str)

    # torchvision中的datasets加载验证集
    validate_dataset = datasets.ImageFolder(root='E:/project helmet/val', transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 打印训练和测试集的个数
    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    # # 训练集图像可视化，可注释
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = test_data_iter.__next__()
    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     print(img.shape)
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # # make_grid的作用是将若干幅图像拼成一幅图像,在需要展示一批数据时很有用
    # imshow(utils.make_grid(test_image))

    efficient_transformer = Linformer(
        dim=128,
        seq_len=49 + 1,  # 7x7 patches + 1 cls-token
        depth=12,
        heads=8,
        k=64
    )

    # Visual Transformer
    model = ViT(
        dim=128,
        image_size=224,
        patch_size=32,
        num_classes=4,
        transformer=efficient_transformer,
        channels=3,
    ).to(device)

    # num_classes=分类个数 init_weights=初始化权重

    loss_function = nn.CrossEntropyLoss()  # 多分类常用的损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0002)  # 优化器
    best_acc = 0.0  # 更新准确率最高的数值
    best_loss = 1.0  # 更新损失最低的数值
    train_steps = len(train_loader)

    epochs = 2
    for epoch in range(epochs):
        # 通过net.train()可以保证dropout/BatchNormal只在训练时候起作用
        model.train()  # model.eval()，不启用 BatchNormalization 和 Dropout。
        # 此时pytorch会自动把BN和DropOut固定住，不会取平均，而是用训练好的值。不然的话，一旦test的batch_size过小，很容易就会因BN层导致模型performance
        # 损失较大；在模型测试阶段使用model.train() 让model变成训练模式，此时 dropout和batch normalization的操作在训练q起到防止网络过拟合的问题。

        running_loss = 0.0  # 统计训练过程中的损失
        train_bar = tqdm(train_loader)  # 显示进度条

        # 训练
        for step, data in enumerate(train_bar):  # data里面是以batch_size为大小的tensor以及这些tensor对应的标签(即分类), step为迭代进度条
            # images: (batchsize,3,224,224)
            # labels: batchsize
            images, labels = data  # 把数据放入images和labels里
            optimizer.zero_grad()  # 清空过往梯度
            # outputs: (batchsize,classes)
            outputs = model(images.to(device))
            loss = loss_function(outputs, labels.to(device))  # 计算预测值与真实值
            loss.backward()  # 损失反向传播
            optimizer.step()  # 更新参数

            # 计算一共多少损失
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss) # 定义进度条的前缀

        # 验证
        model.eval()
        acc = 0.0  # 计算精度公式 number / epoch
        with torch.no_grad():  # 进止pytorch对参数跟踪
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))  # 把测试集的数据传进net中进行对y_predict的求解
                predict_y = torch.max(outputs, dim=1)[1] # 求最大概率的是哪个类别，即预测y
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                # 对两个张量Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True；若不同，返回False，即1或0

        val_accurate = acc / val_num # 计算精度公式 number / epoch

        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' % (epoch + 1, running_loss / train_steps, val_accurate))
        val_loss = running_loss / train_steps
        if val_loss < best_loss:
            best_loss = val_loss
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(model.state_dict(), './epoch%d_train_loss_%.2fval_accuracy_%.2f.pth'
                           % (epoch + 1, running_loss / train_steps, val_accurate))

    print('Finished Training')


if __name__ == '__main__':
    main()

