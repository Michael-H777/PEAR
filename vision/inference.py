import torch 
import torchvision.models as models 
import numpy as np 
import cv2 


def load_model(path): 
    model = models.convnext_base()
    model.load_state_dict(torch.load(path))
    model.cuda()
    return model 


def forward(model, image): 
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
    image = (image-127.2) / np.std(list(range(256)))
    
    data = np.zeros((1, 3, 256, 256), dtype=np.float32)
    data[0,0] = image[:,:,0]
    data[0,1] = image[:,:,1]
    data[0,2] = image[:,:,2]
    
    data = torch.from_numpy(data).cuda()
    output = model(data)
    prediction = torch.argmax(output)[0]
    return prediction
