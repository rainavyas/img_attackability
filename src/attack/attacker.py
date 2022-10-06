import torch

class Attacker():

    @staticmethod
    def can_fgsm(model, x, y, criterion, delta):
        '''Return True if fgsm attack successful within constraint'''
        model.eval()
        
        x.retain_grad()
        y_pred = model(torch.unsqueeze(x, 0))
        loss = criterion(y_pred, torch.unsqueeze(y))
        loss.backward()
        direction = x.grad
        pert = direction/torch.norm(direction)
        x_attack = x+(delta*pert)
        with torch.no_grad():
            y_pred_attack = model(x_attack)
        if torch.argmax(y_pred.squeeze(0)) == torch.argmax(y_pred_attack.squeeze(0)):
            return True
        else:
            return False
    
    @classmethod
    def get_pert_size(cls, model, method='fgsm', min_size=0, max_size=0.2):
        '''
        Find smallest perturbation size required to change prediction of model for sample x
        '''
        # I should implement binary search between smallest and largest values

    @classmethod
    def get_all_pert_sizes(cls, ds, model, method='fgsm', min_size=0, max_size=0.2):
        '''
        Calculate smallest perturbation for adv attack per sample
        '''

        

    
        