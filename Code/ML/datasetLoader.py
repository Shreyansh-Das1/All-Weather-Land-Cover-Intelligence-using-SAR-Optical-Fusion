from torch.utils.data import Dataset
import os
import numpy as np

class datasetLoaderClass(Dataset):
    def __init__(self, files_list):
        self.opt = r"D:\Resume Projs\All-Weather-Land-Cover-Intelligence-using-SAR-Optical-Fusion\data\dataset\optical"
        self.sar = r"D:\Resume Projs\All-Weather-Land-Cover-Intelligence-using-SAR-Optical-Fusion\data\dataset\sar"
        self.label = r"D:\Resume Projs\All-Weather-Land-Cover-Intelligence-using-SAR-Optical-Fusion\data\dataset\labels"
        self.file_list = files_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        sarfile = os.path.join(self.sar,"sar_"+filename)
        optfile = os.path.join(self.opt, "opt_"+filename)
        sar = np.load(sarfile)
        opt = np.load(optfile)
        merged = np.concatenate([sar,opt], 0)
        labels = np.load(os.path.join(self.label,"lbl_"+filename))
        return (merged.astype(np.float32), labels.astype(np.int64))