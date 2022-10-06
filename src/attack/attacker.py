import torch
import numpy as np

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

        loss = torch.log(y_pred[y])
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
    def get_pert_size(cls, x, y, model, device, method='fgsm', min_size=0.02, max_size=0.4, num=40):
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
    def get_all_pert_sizes(cls, ds, model, device, method='fgsm', min_size=0.02, max_size=0.4, num=40):
        '''
        Calculate smallest perturbation for adv attack per sample
        '''
        min_perts = []
        for i in range(len(ds)):
            print(f'On attack {i}/{len(ds)}')
            (x, y) = ds[i]
            min_perts.append(cls.get_pert_size(x, y, model, device, method=method, min_size=min_size, max_size=max_size, num=num))
        return min_perts



        

    
        