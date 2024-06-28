import sys,os
from multiprocessing import cpu_count
import torch

class Config:
    def __init__(self):
        self.default_data_path_name = "data"
        self.sovits_path = ""
        self.gpt_path = ""
        self.cnhubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        self.bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        self.pretrained_sovits_2g_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
        self.pretrained_sovits_2d_path = "GPT_SoVITS/pretrained_models/s2D488k.pth"
        self.pretrained_gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
        self.exp_root = "logs"
        self.python_exec = sys.executable or "python"
        self.is_share_str = os.environ.get("is_share","False")
        self.is_share= True if self.is_share_str.lower() == 'true' else False
        self.webui_port_main = 9874
        self.webui_port_uvr5 = 9873
        self.webui_port_infer_tts = 9872
        self.webui_port_subfix = 9871
        self.api_port = 9880
        self.n_cpu = cpu_count()
        self.ngpu = torch.cuda.device_count()
        self.sovits_weight_root = "SoVITS_weights"
        self.gpt_weight_root = "GPT_weights"
        os.makedirs(self.sovits_weight_root,exist_ok=True)
        os.makedirs(self.gpt_weight_root,exist_ok=True)
        self.is_half_str = os.environ.get("is_half", "True")
        self.is_half = True if self.is_half_str.lower() == 'true' else False
        if torch.cuda.is_available():
            self.infer_device = "cuda"
        else:
            self.infer_device = "cpu"
        print('infer_device:' + self.infer_device)
        if self.infer_device == "cuda":
            self.gpu_name = torch.cuda.get_device_name(0)
            if (
                    ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                    or "P40" in self.gpu_name.upper()
                    or "P10" in self.gpu_name.upper()
                    or "1060" in self.gpu_name
                    or "1070" in self.gpu_name
                    or "1080" in self.gpu_name
            ):
                self.is_half=False
            print('gpu name:' + self.gpu_name)
        if(self.infer_device=="cpu"):self.is_half=False
        gpu_infos = []
        mem = []
        if_gpu_ok = False
        # 判断是否有能用来训练和加速推理的N卡
        if torch.cuda.is_available() or self.ngpu != 0:
            for i in range(self.ngpu):
                gpu_name = torch.cuda.get_device_name(i)
                if any(value in gpu_name.upper()for value in ["10","16","20","30","40","A2","A3","A4","P4","A50","500","A60","70","80","90","M4","T4","TITAN","L4","4060"]):
                    # A10#A100#V100#A40#P40#M40#K80#A4500
                    if_gpu_ok = True  # 至少有一张能用的N卡
                    gpu_infos.append("%s\t%s" % (i, gpu_name))
                    mem.append(int(torch.cuda.get_device_properties(i).total_memory/ 1024/ 1024/ 1024+ 0.4))
        # # 判断是否支持mps加速
        # if torch.backends.mps.is_available():
        #     if_gpu_ok = True
        #     gpu_infos.append("%s\t%s" % ("0", "Apple GPU"))
        #     mem.append(psutil.virtual_memory().total/ 1024 / 1024 / 1024) # 实测使用系统内存作为显存不会爆显存

        if if_gpu_ok and len(gpu_infos) > 0:
            self.gpu_info = "\n".join(gpu_infos)
            self.default_batch_size = min(mem) // 2
        else:
            self.gpu_info = ("%s\t%s" % ("0", "CPU"))
            gpu_infos.append("%s\t%s" % ("0", "CPU"))
            self.default_batch_size = int(psutil.virtual_memory().total/ 1024 / 1024 / 1024 / 2)
        self.gpus="-".join([i[0] for i in gpu_infos])
    # SoVITS模型列表
    def get_sovits_weights_names(self):
        SoVITS_names = [self.pretrained_sovits_2g_path]
        for name in os.listdir(self.sovits_weight_root):
            if name.endswith(".pth"):SoVITS_names.append(name)
        return SoVITS_names
    # GPT模型列表
    def get_gpt_weights_names(self):
        GPT_names = [self.pretrained_gpt_path]
        for name in os.listdir(self.gpt_weight_root):
            if name.endswith(".ckpt"): GPT_names.append(name)
        return GPT_names  
