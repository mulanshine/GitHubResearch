def create_model(opt):
    model = None
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'c_gan':
        assert(opt.dataset_mode == 'segmentation')
        from .c_gan_model import CGANModel
        model = CGANModel()
    elif opt.model == 'gan':
        assert(opt.dataset_mode == 'segmentation')
        from .gan_model import GANModel
        model = GANModel()
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model