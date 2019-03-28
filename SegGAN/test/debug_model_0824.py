import time
import torch
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from torchvision.models.vgg import VGG
from models import networks

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
opt.gpu_ids = [3,2]
opt.batchSize = 1
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)
model.netD.module.load_state_dict(torch.load("/home/jjchu/My_Research/SegGAN/checkpoints/experiment_cgan_strided2/10_net_D.pth"))
model.netG.module.load_state_dict(torch.load("/home/jjchu/My_Research/SegGAN/checkpoints/experiment_cgan_strided2/10_net_G.pth"))

visualizer = Visualizer(opt)
for i, data1 in enumerate(dataset):
    data=data1
    if i == 10:
        break
    visualizer.reset()
    model.set_input(data)
    model.optimize_parameters()
    visualizer.display_current_results(model.get_current_visuals(), 1, 1)
    losses = model.get_current_losses()
    visualizer.print_current_losses(1, 2, losses, 1, 2)
    print('pred_fake: %f, pred_real: %f' % (model.pred_fake[0], model.pred_real[0]))
    print('real_label:')
    print(model.real_label[0])
    print('pred_fake_cls:')
    print(model.pred_fake_cls[0])
    print('pred_real_cls:')
    print(model.pred_real_cls[0])

# vggmodel = VGG(networks.make_layers(opt.cfg, batch_norm=True))
# vggmodel.load_state_dict(torch.load(opt.vgg16bn_pre_model))
# pretrained_dict = vggmodel.state_dict()
# model_dict = model.netD.module.state_dict()
# pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# model_dict.update(pretrained_dict)
# model.netD.module.load_state_dict(model_dict)
# visualizer.plot_current_losses(1, float(2) / 1, opt, losses)
