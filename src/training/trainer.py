import torch 
from..tools.tools import AverageMeter, accuracy_topk


class Trainer():
    '''
    All training functionality
    '''
    def __init__(self, device, model, optimizer, criterion, scheduler):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
    
    @staticmethod
    def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=25):
        '''
        Run one train epoch
        '''
        losses = AverageMeter()
        accs = AverageMeter()

        # switch to train mode
        model.train()

        for i, (x, y) in enumerate(train_loader):

            x = x.to(device)
            y = y.to(device)
            # Forward pass
            logits = model(x)
            loss = criterion(logits, y)

            # Backward pass and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            acc = accuracy_topk(logits.data, y)
            accs.update(acc.item(), x.size(0))
            losses.update(loss.item(), x.size(0))

            if i % print_freq == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\tLoss {losses.val:.4f} ({losses.avg:.4f})\tAccuracy {accs.val:.3f} ({accs.avg:.3f})')


    @staticmethod
    def eval(val_loader, model, criterion, device, return_logits=False):
        '''
        Run evaluation
        '''
        losses = AverageMeter()
        accs = AverageMeter()

        # switch to eval mode
        model.eval()

        all_logits = []
        with torch.no_grad():
            for (x, y) in val_loader:

                x = x.to(device)
                y = y.to(device)

                # Forward pass
                logits = model(x)
                all_logits.append(logits)
                loss = criterion(logits, y)

                # measure accuracy and record loss
                acc = accuracy_topk(logits.data, y)
                accs.update(acc.item(), x.size(0))
                losses.update(loss.item(), x.size(0))

        if return_logits:
            return torch.cat(all_logits, dim=0).detach().cpu()

        print(f'Test\t Loss ({losses.avg:.4f})\tAccuracy ({accs.avg:.3f})\n')
        return accs.avg

    
    def train_process(self, train_dl, val_dl, save_path, max_epochs=300):

        best_acc = 0
        for epoch in range(max_epochs):

            # train for one epoch
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            self.train(train_dl, self.model, self.criterion, self.optimizer, epoch, self.device)
            self.scheduler.step()

            # evaluate on validation set
            acc = self.eval(val_dl, self.model, self.criterion, self.device)
            if acc > best_acc:
                best_acc = acc
                state = self.model.state_dict()
                torch.save(state, save_path)