import torch
import numpy as np
from itertools import product

class Attacker():
    @staticmethod
    def gradient(model, x, y, device):
        '''
        Return gradient of loss wrt to input x
        '''
        x = x.to(device)
        model.eval()

        x.requires_grad = True
        y_pred = model(torch.unsqueeze(x, 0)).squeeze(0)

        loss = -1*torch.log(y_pred[y])
        loss.backward()
        direction = x.grad
        return y_pred.squeeze(0).cpu().detach(), direction.cpu().detach(),

    @staticmethod
    def can_fgsm(model, x, y_pred, direction, delta, device):
        '''Return True if fgsm attack successful within constraint (l_inf norm)'''
        model.eval()
        sign = torch.sign(direction)
        x_attack = x+(delta*sign)
        x_attack = x_attack.to(device)
        with torch.no_grad():
            y_pred_attack = model(torch.unsqueeze(x_attack, 0)).squeeze(0)
        if torch.argmax(y_pred).item() == torch.argmax(y_pred_attack).item():
            return False
        else:
            return True
    
    @classmethod
    def get_pert_size(cls, x, y, model, device, method='fgsm', min_size=0.02, max_size=0.4, num=100):
        '''
        Find smallest perturbation size required to change prediction of model for sample x
        If all sizes fail, returns max_size
        '''
        y_pred, direction = cls.gradient(model, x, y, device)

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
    def get_all_pert_sizes(cls, ds, model, device, method='fgsm', min_size=0.02, max_size=0.4, num=100):
        '''
        Calculate smallest perturbation for adv attack per sample
        '''
        min_perts = []
        for i in range(len(ds)):
            print(f'On attack {i}/{len(ds)}')
            (x, y) = ds[i]
            min_perts.append(cls.get_pert_size(x, y, model, device, method=method, min_size=min_size, max_size=max_size, num=num))
        return min_perts
    
    @staticmethod
    def attack_frac_sweep(perts, start=0.0, end=0.38, num=100):
        '''
        Return fraction of attackable samples at each perturbation size threshold
        '''
        threshs = np.linspace(start, end, num)
        size = len(perts)
        frac_attackable = []
        for t in threshs:
            num_att = len([p for p in perts if p<=t])
            frac_attackable.append(num_att/size)
        return threshs, frac_attackable

    @staticmethod
    def attack_frac_sweep_all(perts_all, start=0.0, end=0.38, num=100):
        '''
        Return fraction of attackable samples (over all models) at each perturbation size threshold
        '''
        threshs = np.linspace(start, end, num)
        size = len(perts_all[0])
        frac_attackable = []
        for t in threshs:
            num_att = 0
            for sample in zip(*perts_all):
                smaller = True
                for pert in sample:
                    if pert > t:
                        smaller = False
                        break
                if smaller:
                    num_att+=1
            frac_attackable.append(num_att/size)
        return threshs, frac_attackable
    




        

    
        