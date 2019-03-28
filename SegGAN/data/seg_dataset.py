import os.path as osp
import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_labels, make_img_name_list, make_seg_vocdataset
from PIL import Image

class SegImageDataset(BaseDataset):
    """
        return {'image':image,'paths':path,'label':label}
    """ 
    def initialize(self, opt):
        self.opt = opt 
        self.labels_path = os.path.join(opt.cls_labels)  # cls_label.npy: name : [0,0,1,0,...,0]
        self.dimg_labels = make_labels(self.labels_path)   # dict: numpy array  'name' : [0,0,1,0,...,0]

        self.root = opt.dataroot  
        self.data_txt = opt.data_txt
        self.img_list_path = osp.join(self.root,'ImageSets','Segmentation', str(self.data_txt)+'.txt')
        self.img_name_list = make_img_name_list(self.img_list_path)
        self.dimg_path = make_seg_vocdataset(self.root, self.img_name_list, self.data_txt)  # dict
        self.transform = get_transform(opt)

    def __getitem__(self, index):
        img_name = self.img_name_list[index]
        path = self.dimg_path[img_name]
        label = self.dimg_labels[img_name]
        img = Image.open(path).convert('RGB')
        # print(img.size)
        # print(path)
        image = self.transform(img)
        input_nc = self.opt.input_nc
        if input_nc == 1:  # RGB to gray
            tmp = image[0, ...] * 0.299 + image[1, ...] * 0.587 + image[2, ...] * 0.114
            image = tmp.unsqueeze(0)
        return {'image':image,'paths':path,'label':label}

    def __len__(self):
        return len(self.img_name_list)

    def name(self):
        return 'SegImageDataset'
