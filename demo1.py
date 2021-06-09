import numpy as np
import torch
import jittor as jt

jt.flags.use_cuda = 0
if __name__ == '__main__':
    np.random.seed(648)
    x = np.random.uniform(0, 1, (2, 3, 224, 224)).astype(np.float32)
    x1 = jt.array(x)
    m1=jt.nn.BatchNorm2d(3)
    x1 = m1(x1).data

    x2 = torch.tensor(x)
    m2=torch.nn.BatchNorm2d(3)
    x2 = m2(x2).detach().numpy()

    print(np.sum(np.abs(x1 - x2)))
