from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import montage
from keras.models import load_model
model = load_model('19_january_150k.h5',compile=False)
import skimage 
from keras.preprocessing import image
import cv2
from django.core.files.storage import FileSystemStorage
def index(request):
    context={'a':1}
    return render(request,'index.html',context)
def realImage(request):
    print(request)
    print(request.POST.dict())
    fileobj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileobj.name,fileobj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    i = image.load_img(testimage)
    i = skimage.img_as_float(i)
    i = image.img_to_array(i)
    i = np.expand_dims(i, axis = 0)
    p = model.predict(i)
    montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
    # plt.imshow(filePathName)
    plt.imshow(montage(p[:, :, :, 0]), cmap = 'Reds',alpha=1)
    plt.axis('off')
    plt.savefig('media/pred/'+fileobj.name+'.jpg',bbox_inches='tight', pad_inches=0)
    context={'filePathName':filePathName,'rr':'media/pred/'+fileobj.name+'.jpg'}
    return render(request,'index.html',context)

# def predictImage(request):
#     print(request)
#     print(request.POST.dict())
#     fileobj=request.FILES['filePath']
#     fs=FileSystemStorage()
#     filePathName=fs.save(fileobj.name,fileobj)
#     filePathName=fs.url(filePathName)
#     testimage='.'+filePathName
#     img_width, img_height = 300, 300
#     fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize = (24, 8))
#     i = image.load_img(testimage, target_size = (img_width, img_height))
#     ax1.imshow(i,alpha=1)
#     ax1.set_title('Actual Image')
#     ax2.imshow(i,alpha=1)
#     i = skimage.img_as_float(i)
#     i = image.img_to_array(i)
#     i = np.expand_dims(i, axis = 0)
#     p = model.predict(i)
#     print(p)
#     print('p', p.shape, p.dtype, p.min(), p.max())
#     montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
#     ax2.imshow(montage(p[:, :, :, 0]), cmap = 'Reds',alpha=0.25)
#     ax2.set_title('Predicted Image overlayed on Actual Image')
#     ax3.imshow(montage(p[:, :, :, 0]), cmap = 'Reds',alpha=1)
#     context={'filePathName':ax3}
#     # filePathNamecontext={'a':1}
#     return render(request,'index.html',context)