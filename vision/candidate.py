import torch
from itertools import cycle 
from torchvision.models import convnext_base

class single_task:
    
    def __init__(self, *, name, ddp, steps):
        self.name = self.__class__.__name__ if name is None else name

        self.update_at_step = 64
        self.step_counter = cycle(list(range(1, self.update_at_step+1)))
        
        model = convnext_base(pretrained=True)
        if ddp: 
            self.model = torch.nn.parallel.DistributedDataParallel(model.cuda())
        else:
            self.model = torch.nn.DataParallel(model).cuda()
        self.loss = torch.nn.CrossEntropyLoss() 
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scaler = torch.cuda.amp.GradScaler()
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-2, total_steps=steps//self.update_at_step, pct_start=0.2)
        
    def train(self):
        with torch.cuda.amp.autocast(): 
            output = self.model(self.input_data)
        loss = self.loss(output, self.target)
        self.scaler.scale(loss).backward() 
        
        if next(self.step_counter) == self.update_at_step:
            self.scaler.step(self.optimizer)
            self.scaler.update() 
            self.scheduler.step() 
            self.optimizer.zero_grad()
        return self.current_lr

    @torch.no_grad()
    def test(self):
        with torch.cuda.amp.autocast(): 
            logits = self.model(self.input_data)
        predictions = torch.argmax(logits, dim=1)
        acc = (predictions==self.target).sum() / predictions.shape[0] 
        return acc 

    def set_input(self, data):
        self.input_data = data[0].cuda()
        self.target = data[1].long().cuda()
        return None

