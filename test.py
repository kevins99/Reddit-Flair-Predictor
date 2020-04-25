from cf import centerface
from model_67 import SFAnet
import torch
from torchvision import datasets, transforms
import cv2
import PIL
import numpy as np

import sys

model=centerface()
# ck=torch.load("bias_checkpoint.pth.tar")
ck=torch.load(sys.argv[1])

model.load_state_dict(ck['state_dict'])
model.eval()
model.cuda()
# cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
viz=cv2.resize(cv2.imread("0_Parade_marchingband_1_873.jpg"),(800,800))
# 
img=np.array(cv2.resize(cv2.cvtColor(cv2.imread("0_Parade_marchingband_1_873.jpg"),cv2.COLOR_BGR2RGB),(800,800)),dtype=np.float32)

print(img.dtype,"qweqew")
# img=PIL.Image.fromarray(img)
# img=torch.from_numpy(np.transpose(np.array(img),(2,0,1))).float()
# print(img)
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

img = transform(img)
print(img.shape)
# img=np.transpose(img,(2,0,1))
img=img.unsqueeze(0)
print(img.dtype)
#img=torch.randn((1,3,800,800))

img=img.float().cuda()
cv2.imshow("test",np.transpose(np.array(img.cpu().detach().numpy()[0]),(1,2,0)).astype(np.uint8))
h,s,o = model(img) #0,s
print(h.shape)
print((np.where(h.cpu().detach()>=0.5)))
print(np.where(h.cpu().detach()[0,0]>=0.1))
conf=h.cpu().detach()[0,0][np.where(h.cpu().detach()[0,0]>=0.2)]
conf=zip(list(conf),np.where(h.cpu().detach()[0,0]>=0.2))
print(conf,"qwe")
conf=sorted(conf,key=lambda x: x[0])
print(o.cpu().detach()[0,1][np.where(h.cpu().detach()[0,0]>=0.5)],"size")
print(np.sum(h.cpu().detach().numpy()[0,0]))
temp=np.array(h.cpu().detach()[0,0])
temp[np.where(h.cpu().detach()[0,0]>=0.5)]=1
temp[np.where(h.cpu().detach()[0,0]<0.5)]=0
cv2.imshow("asd",h.cpu().detach()[0,0].numpy())
cv2.imshow("test123",temp)
cv2.waitKey(0)
