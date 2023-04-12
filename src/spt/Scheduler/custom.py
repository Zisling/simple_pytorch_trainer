from torch.optim.lr_scheduler import LambdaLR


def create_lambda_scheduler(lr_lambda):
    return lambda optimizer: LambdaLR(optimizer, lr_lambda)


