from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import torch
# from options.test_options import TestOptions
from options.train_options import TrainOptions
from torch import is_tensor
import random
import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer

RANDOM_SEED = 1445754
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

def tensor2im(input_image, imtype=np.uint8):
    if is_tensor(input_image):
        image_tensor = input_image
    elif isinstance(input_image, Variable):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

def saveFig(feature_conv,imgpath):
    feature_conv = tensor2im(feature_conv)
    cv2.imwrite(imgpath,feature_conv)

def save_generate_fig(fake_img,fakepath):
    # feature_conv = torch.from_numpy(feature_conv)
    img = tensor2im(fake_img)
    # feature_conv = feature_conv.numpy()
    cv2.imwrite(fakepath,img)

def write_file(path,mess):
    print(mess)
    with open(path, 'a+') as net_file: # a+
        net_file.write(str(mess))
        net_file.write('\n')

def label2cat(label):
    CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor']
    cat = dict()
    for i in range(20):
        cat[CAT_LIST[i]] = label[i]
    return cat

def vis_features_layers(feature_conv,th=40,t=24,layers=18,set=False):
    size_upsample = (256, 256)
    feature = np.max(feature_conv,axis=1) # (1,256,256)
    # feature = np.mean(feature_conv,axis=1)
    feature = np.squeeze(feature) #(256,256)
    feature = feature - np.min(feature)
    feature = feature/np.max(feature)
    # feature = feature.resize(size_upsample)
    feature = cv2.resize(feature, size_upsample)
    feature = np.uint8(255 * feature)
    for x in range(len(feature[0])):
        for y in range(len(feature[1])):
            if feature[x,y] < th:
                feature[x,y] = 0
            elif set and feature[x,y] >= th:
                feature[x,y] = 255
    # feature = cv2.resize(feature, size_upsample)
    # output.append(cv2.resize(feature, size_upsample))
    heatmap = cv2.applyColorMap(feature, cv2.COLORMAP_JET)
    if set:
        name = './test/result_layers/netG/Down'+ str(t) +'_heat_'+ str(layers) +'.jpg'
    else:
        name = './test/result_layers/netG/Down'+ str(t) +'_heat_'+ str(layers) +'.jpg'
    cv2.imwrite(name, heatmap)

# opt = TestOptions().parse()
opt = TrainOptions().parse()
opt.nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# G_path = "./checkpoints/experiment_name/50_net_G.pth"
# D_path = "./checkpoints/experiment_name/50_net_D.pth"
G_path = "./checkpoints/experiment_softmax_bce_f50/40_net_G.pth"
D_path = "./checkpoints/experiment_softmax_bce_f50/40_net_D.pth"

net = create_model(opt)
# write_file(file_path, net)
net.netG.module.load_state_dict(torch.load(G_path))
net.netD.module.load_state_dict(torch.load(D_path))
th = 0
t = 1
# for t in range(1,10):
imgpath = './test/result_layers/netG/' + str(t) + 'real.jpg'
fakepath = './test/result_layers/netG/' + str(t) + 'fake.jpg'
# layers = [1,3,4,6,8,9,11,13,15]
# for layer in layers:
layer=6
features_blobs = []
net.netG.module.model[0].Downsample[layer].register_forward_hook(hook_feature)
# net.netG.module.model[layer].register_forward_hook(hook_feature)
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
for i, data1 in enumerate(dataset):
    if i == t:
        data = data1
        break

real_img = data['image'].float()
real_label = data['label'].float() 
if len(opt.gpu_ids) > 0:
    real_img = real_img.cuda(opt.gpu_ids[0], async=True)
    real_label = real_label.cuda(opt.gpu_ids[0], async=True)


image_paths = data['paths'] 
real_img = Variable(real_img) 
real_label= Variable(real_label)
fake_img = net.netG(real_img,real_label) # (1,3,256,256)
pred_real,pred_real_label = net.netD(real_img)
pred_fake,pred_fake_label = net.netD(fake_img)
# print(label2cat(pred_real_label[0]))
# print(label2cat(pred_fake_label[0]))
print(label2cat(pred_real_label.data.cpu().numpy()[0]))
print(label2cat(real_label.data.cpu().numpy()[0]))
save_generate_fig(fake_img,fakepath)
saveFig(real_img,imgpath)
feature_conv = features_blobs[0] # (1,3,256,256)
img = cv2.imread(imgpath)
height, width, _ = img.shape
MAPs = vis_features_layers(feature_conv,th,t,layer,False)



# MAPs = vis_heat_test(feature_conv,imgpath,th,t,layers,False)
# for th in [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]:
#     MAPs = vis_heat_test(feature_conv,imgpath,th,False)
    # MAPs = vis_heat_test(feature_conv,imgpath,th,True)
# # saveFig(feature_conv)
# # img_pil = Image.open(image_paths[0])
# # test_name = './result/result26/test25.jpg'
# # img_pil.save(test_name)
# # MAPs = vis_Threshold(feature_conv,th,True)
# # MAPs = vis_Threshold(feature_conv,th,False)
# MAPs = vis_heat_test(feature_conv,imgpath,0,False)
# # MAPs = vis_max(features_blobs[0])
# # MAPs = vis_mean(features_blobs[0])
# print(MAPs)
# # vis_channels(feature_conv)
# # img = mpimg.imread(image_paths[0])
# heatmap = cv2.applyColorMap(cv2.resize(MAPs[0],(width, height)), cv2.COLORMAP_JET)
# # # heatmap = cv2.applyColorMap(cv2.resize(MAPs[0],(width, height)), cv2.COLORMAP_AUTUMN)
# # cv2.imwrite('./result/result_th/heatmap_26.jpg', heatmap)
# # result = heatmap * 0.3 + img * 0.5
# # cv2.imwrite('./result/result_th/CAM_26.jpg', result)
# # heatname = './result_GAP/result/' + str(t) + 'heatmap_'+ str(layers) + '_' + str(th) +'.jpg'
# # camname = './result_GAP/result/' + str(t) + 'CAM_'+ str(layers) + '_' + str(th) +'.jpg'
# # heatname = './result_GMP/result_th/' + str(t) + 'heatmap_'+ str(layers) + '_' + str(th) + '.jpg'
# # camname = './result_GMP/result_th/' + str(t) + 'CAM_'+ str(layers) + '_' + str(th) + '.jpg'
# name = './result_GMP/test_heat_test.jpg'
# cv2.imwrite(name,heatmap)
# # result = heatmap * 0.3 + img * 0.5
# # cv2.imwrite(camname, result)