from tensorboardX import SummaryWriter

writer = SummaryWriter("./logs")
for i in range(10):
    writer.add_scalar("y=x",i,i)
writer.close()
