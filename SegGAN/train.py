import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from torchvision.models.vgg import VGG
from models import networks
import torch

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)
    model = create_model(opt)
    networks.write_net_file(opt, model)
    # model.netD.module.load_state_dict(torch.load("/home/jjchu/My_Research/SegGAN/checkpoints/experiment_dialedNL100/100_net_D.pth"))
    # model.netG.module.load_state_dict(torch.load("/home/jjchu/My_Research/SegGAN/checkpoints/experiment_dialedNL100/100_net_G.pth"))
    # load parameters from pretrained vgg16
    # vggmodel = VGG(networks.make_layers(opt.cfg, batch_norm=True))
    # vggmodel.load_state_dict(torch.load(opt.vgg16bn_pre_model))
    # model = create_model(opt)
    # pretrained_dict = vggmodel.state_dict()
    # model_dict = model.netD.module.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.netD.module.load_state_dict(model_dict)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1): # +1
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            # model.D_optimize_parameters()
            # model.G_optimize_parameters()
            model.optimize_parameters()
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batchSize
                # visualizer.print_current_losses(1, 1, losses, 2, 5)
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    # visualizer.plot_current_losses(1, float(1) / 1, opt, losses)
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)


            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('latest')

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)
            print('pred_fake: %f, pred_real: %f' % (model.pred_fake[0], model.pred_real[0]))
            print('real_label:')
            print(model.real_label[0])
            print('pred_fake_cls:')
            print(model.pred_fake_cls[0])
            print('pred_real_cls:')
            print(model.pred_real_cls[0])

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
