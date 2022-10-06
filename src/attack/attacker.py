import torch
import numpy as np

class Attacker():
    @staticmethod
    def gradient(model, x, y, criterion, device):
        '''
        Return gradient of loss wrt to input x
        '''
        x = x.to(device)
        y = torch.LongTensor(y).to(device)
        model.eval()

        x.requires_grad = True
        y_pred = model(torch.unsqueeze(x, 0))
        loss = criterion(y_pred, torch.unsqueeze(y))
        loss.backward()
        direction = x.grad
        return y_pred.squeeze(0).cpu().detach(), direction.squeeze(0).cpu().detach(),

    @staticmethod
    def can_fgsm(model, x, y_pred, direction, delta, device):
        '''Return True if fgsm attack successful within constraint (l_inf norm)'''
        model.eval()
        sign = torch.sign(direction)
        x_attack = x+(delta*sign)
        x_attack = x_attack.to(device)
        with torch.no_grad():
            y_pred_attack = model(torch.unsqueeze(x_attack, 0)).squeeze(0)
        if torch.argmax(y_pred.item()) == torch.argmax(y_pred_attack.item()):
            return False
        else:
            return True
    
    @classmethod
    def get_pert_size(cls, x, y, model, criterion, device, method='fgsm', min_size=0.02, max_size=0.3, num=20):
        '''
        Find smallest perturbation size required to change prediction of model for sample x
        If all sizes fail, returns max_size
        '''
        y_pred, direction = cls.gradient(model, x, y, criterion, device)

        # binary search for smallest perturbation size
        deltas = np.linspace(min_size, max_size, num)
        l = 0
        r = len(deltas) - 1
        while r-l>0:
            if r-l == 1:
                if cls.can_fgsm(model, x, y_pred, direction, deltas[l], device):
                    return deltas[l]
                else:
                    return deltas[r]
            mid = int((r+l)/2)
            if cls.can_fgsm(model, x, y_pred, direction, deltas[mid], device):
                 r = mid
            else:
                l = mid
        return deltas[r]


    @classmethod
    def get_all_pert_sizes(cls, ds, model, criterion, device, method='fgsm', min_size=0.02, max_size=0.3, num=20):
        '''
        Calculate smallest perturbation for adv attack per sample
        '''
        min_perts = []
        for i in range(len(ds)):
            print(f'On attack {i}/{len(ds)}')
            (x, y) = ds[i]
            min_perts.append(cls.get_pert_size(x, y, model, criterion, device, method=method, min_size=min_size, max_size=max_size, num=num))
        return min_perts



        

    
        