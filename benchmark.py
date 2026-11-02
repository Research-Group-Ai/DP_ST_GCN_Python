import argparse,time,torch
from model import DPSTGCN
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--iterations',type=int,default=100);a=p.parse_args()
    device='cuda' if torch.cuda.is_available() else 'cpu';model=DPSTGCN().to(device).eval();x=torch.randn(1,3,100,50,device=device)
    with torch.no_grad():
        for _ in range(20):model(x)
        if device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        for _ in range(a.iterations):model(x)
        if device=='cuda':torch.cuda.synchronize()
    print('Parameters:',sum(p.numel() for p in model.parameters()));print('Mean milliseconds per clip:',1000*(time.perf_counter()-start)/a.iterations)
