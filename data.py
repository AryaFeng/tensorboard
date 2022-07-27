
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms
from tensorboardX import SummaryWriter

train_set = FashionMNIST("./data",True,transform =transforms.ToTensor() ,download=True)

train_loader = DataLoader(dataset=train_set,batch_size=64,shuffle=True)
writer = SummaryWriter("./logs")

# print(len(train_loader))
global_step = 0
for images,labels in train_loader:
    writer.add_images("images",images,global_step=global_step)
    global_step+=1
writer.close()