"""train

This module is homework1 for data mining class.
Main process in this module is visualizing MNIST
dataset then add some customized data and train 
it, finally plot results.

the following comment has nothing to do with the code
but to record the process of experiment as to write into
homework report.
实验过程：
1. 去掉了crop和flip等data augmentation，因为部分数据会被裁掉
2. 修改了resnet第一层以适应BW格式图片
3. 使用没有预训练的resnet模型
"""

from __future__ import barry_as_FLUFL

__version__ = '1.0'
__author__ = 'Haoyu Qi'

# import necessary packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import torchvision
from torchvision import transforms,utils,models
import argparse
import seaborn as sns
from collections import Counter
import time
import copy
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image
from pathlib import Path

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def dataVisualization(x_train,y_train,x_test,y_test):
    '''return null

    visualing first nine images
    save it as a image
    then visualize distribution of training set
    save it as a image
    '''
    plt.figure(figsize=(9, 9))
    for i in range(9):
        plt.subplot(331 + i)
        plt.imshow(x_train[i], cmap=plt.get_cmap('gray'))
    plt.savefig('sample.png', bbox_inches='tight', dpi=300)
    print(f'训练集的样本数：{x_train.shape[0]}，测试集的样本数：{x_test.shape[0]}')
    print(f'输入图像的大小：{x_train.shape[1]}*{x_train.shape[2]}') 
    y_train = np.array(y_train)
    label_cnt = Counter(y_train)  # 统计每个类别的样本数
    print('训练集的图像类别分布：', label_cnt)
    # 可视化图像类别分布
    plt.figure(figsize=(9,9))
    plt.pie(x=label_cnt.values(), labels=label_cnt.keys(), autopct='%.2f%%')
    plt.savefig('label_distribution.png', bbox_inches='tight', dpi=300)

def train_model(model, dataloaders, writer, criterion, optimizer, scheduler, device, args, num_epochs=25, save=True):
    '''return model

    load input model as last best model, begin trainning for num_epochs
    '''
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            count = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print(inputs.shape)
                # print(labels.shape)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # print(f"outputs done:{outputs.shape}")
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                count += inputs.size(0)

            epoch_loss = running_loss / count
            epoch_acc = running_corrects.double() / count

            if phase == 'train':
                scheduler.step()
                # log the running loss
                writer.add_scalar('training loss',
                            epoch_loss,
                            epoch)
                writer.add_scalar('training acc',
                            epoch_acc,
                            epoch)

            if phase == 'val':
                # log the running loss
                writer.add_scalar('validation loss',
                            epoch_loss,
                            epoch)
                writer.add_scalar('validation acc',
                            epoch_acc,
                            epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    if save:
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_wts,
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': best_acc,
            }, args.SaveModel)
    return model

def test(model,test_loader,device):
    # set model to test mode
    model.eval()
    test_acc = 0.0
    count = 0

    # Initialize the prediction and label lists(tensors)
    predlist=torch.zeros(0,dtype=torch.long, device='cpu')
    lbllist=torch.zeros(0,dtype=torch.long, device='cpu')

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Predict classes using images from the test set
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)

        test_acc += torch.sum(prediction == labels.data).float()
        count += images.shape[0]

        # Append batch prediction results
        predlist=torch.cat([predlist, prediction.view(-1).cpu()])
        lbllist=torch.cat([lbllist, labels.view(-1).cpu()])

    test_acc = test_acc / count
    print(f"total test accuracy:{test_acc} over {count} test samples")

    # Confusion matrix
    conf_mat=confusion_matrix(lbllist.numpy(), predlist.numpy())
    print(conf_mat)

    # Per-class accuracy
    accuracy = accuracy_score(lbllist, predlist)
    return test_acc

def my_model():
    model_body = models.resnet18(pretrained=False)
    
    num_ftrs = model_body.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_body.fc = nn.Linear(num_ftrs, 10)

    # modify first conv to accept bw img
    # weight = copy.deepcopy(model_body.conv1.weight)
    # new_weight = (torch.sum(weight,dim = 1)/3).unsqueeze(1)

    model_body.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model_body.conv1.weight.data = new_weight

    return model_body

def check_model(trainloader,net):
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    images = images.to(device)
    labels = labels.to(device)

    # create grid of images
    img_grid = torchvision.utils.make_grid(images)

    # write to tensorboard
    writer.add_image('four_mnist_images', img_grid)

    writer.add_graph(net, images)
    writer.close()

def stratified_sampling(ds,k):
    '''return trainset,validset

    sampling from ds while maintaining data balance
    acoss different classes.
    k is the number of samples in each class in trainset.
    '''
    class_counts = {}
    train_indices = []
    test_indices = []
    for i, (data, label) in enumerate(ds):
        c = label
        class_counts[c] = class_counts.get(c, 0) + 1
        if class_counts[c] <= k:
            train_indices.append(i)
        else:
            test_indices.append(i)

    print('stratified sampling done')
    return (train_indices,test_indices)

def predict_for_customized_data(model,dataPath,device):
    model.eval()
    for i in range(10):
        img1 = Image.open(dataPath/(str(i)+'.JPG')).convert('L')
        sized_img1 = transforms.Resize((28,28))(img1)
        sized_img1 = 255 - np.array(sized_img1)
        sized_img1[sized_img1<128]=0
        sized_img1[sized_img1>=128]=255
        tensor1 = transforms.ToTensor()(sized_img1)
        tensor1 = transforms.Normalize([0.485], [0.229])(tensor1)
        tensor1 = torch.unsqueeze(tensor1,0)
        assert tensor1.shape == (1,1,28,28)
        inputs = tensor1.to(device)
        outputs = model(inputs)
        # print(outputs)
        _, prediction = torch.max(outputs.data, 1)
        print(prediction)


if __name__ == '__main__':
    # argparse for using this code base
    parser = argparse.ArgumentParser(description="Train a network for MNIST dataset,  \
                                            please make sure your computer has a GPU")
    
    parser.add_argument("datasetPath", help="input path where your datasets are",
                    type=str, nargs='?', default='/home/432/qihaoyu/data/MNIST')

    parser.add_argument("logDir", help="input dir where your logs will be recorded", \
                    type=str, nargs='?', default='/home/432/qihaoyu/vscode_workspace/homework/mnist_experiment_1')

    parser.add_argument('SaveModel', help="input path where your model will be saved", \
                    type=str,nargs='?',default='/home/432/qihaoyu/vscode_workspace/homework/models')
    parser.add_argument("img_size", help="standard image size to feed into the network",
                    type=int, nargs='?', default=224)    

    parser.add_argument("splits", help="ratio for trainning samples in MNIST trainset, \
                            remaining samples will be treated as validset", type=float,
                            nargs='?', default=0.8)       

    args = parser.parse_args() 
    
    # transformation
    data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ]),
    'test': transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229])
    ])
    }
    
    # create tensorboard writer
    writer = SummaryWriter(args.logDir)
    
    # get pytorch datasets
    # if there's no existing datasets in the root directory, you should download it first
    trainset = torchvision.datasets.MNIST(args.datasetPath, train=True, download=False)
    testset = torchvision.datasets.MNIST(args.datasetPath, train=False, download=False)
    
    x_train = trainset.train_data
    y_train = trainset.train_labels
    x_test = testset.test_data
    y_test = testset.test_labels

    # visualize data and distribution according to instructions in ppt
    dataVisualization(x_train, y_train,x_test,y_test)
    
    # transform data and generate data loader to create batches
    raw_trainset = torchvision.datasets.MNIST(args.datasetPath, train=True, download=False,
                                            transform=data_transforms['train'])
    train_indices, valid_indices = stratified_sampling(trainset, 0.8*len(trainset)/10)
    trainset = Subset(raw_trainset,train_indices)
    validset = Subset(raw_trainset,valid_indices)


    
    train_data_loader = DataLoader(trainset,batch_size=128,shuffle=True,drop_last=False)
    valid_data_loader = DataLoader(validset,batch_size=128,shuffle=False,drop_last=False)

    testset = torchvision.datasets.MNIST(args.datasetPath, train=False,download=False,
                                            transform=data_transforms['test'])
    test_data_loader = DataLoader(testset,batch_size=128,shuffle=False,drop_last=False)

    # generate dataloaders dict
    dataloaders = {'train':train_data_loader, 'val':valid_data_loader}
    dataset_sizes = {'train':len(trainset), 'val':len(validset)}

    # if running on GPU and we want to use cuda move model there
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # instantiate network(body:resnet + head:fc)
    model = my_model()
    model = model.to(device)
    check_model(train_data_loader,model)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    model = train_model(model, dataloaders, writer, criterion, optimizer_ft, exp_lr_scheduler,
                        device, args, num_epochs=25,save=True)
    
    # test model
    test(model,test_data_loader,device)

    dataPath = Path('/home/432/qihaoyu/vscode_workspace/homework/imgs')
    predict_for_customized_data(model,dataPath,device)

