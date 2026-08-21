import os
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader

from dataset import DF2KPairedDataset, get_df2k_paths
from architecture import NAFNetSR

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _gk2d(size=11, sigma=1.5, device='cuda', dtype=torch.float32):
    c = torch.arange(size, dtype=dtype, device=device) - (size-1)/2.0
    g = torch.exp(-(c**2)/(2.0*sigma**2))
    k = g.unsqueeze(1)*g.unsqueeze(0)
    return (k/k.sum()).view(1,1,size,size)

@torch.no_grad()
def calculate_metrics_gpu(sr_t, hr_t, scale=4):
    sr = torch.round(torch.clamp(sr_t.float()*255,0,255))
    hr = torch.round(torch.clamp(hr_t.float()*255,0,255))
    if scale>0: sr,hr = sr[:,:,scale:-scale,scale:-scale], hr[:,:,scale:-scale,scale:-scale]
    sr_y = 16.0+(65.481*sr[:,0:1]+128.553*sr[:,1:2]+24.966*sr[:,2:3])/255.0
    hr_y = 16.0+(65.481*hr[:,0:1]+128.553*hr[:,1:2]+24.966*hr[:,2:3])/255.0
    mse = torch.mean((sr_y-hr_y)**2).item()
    psnr = float('inf') if mse==0 else 10.0*math.log10(255.0*255.0/mse)
    k = _gk2d(11,1.5,sr_y.device,sr_y.dtype)
    C1,C2 = (0.01*255)**2, (0.03*255)**2
    mu1,mu2 = F.conv2d(sr_y,k,padding=0), F.conv2d(hr_y,k,padding=0)
    mu1s,mu2s,mu12 = mu1**2, mu2**2, mu1*mu2
    s1s = F.conv2d(sr_y**2,k,padding=0)-mu1s
    s2s = F.conv2d(hr_y**2,k,padding=0)-mu2s
    s12 = F.conv2d(sr_y*hr_y,k,padding=0)-mu12
    ssim = ((2*mu12+C1)*(2*s12+C2)/((mu1s+mu2s+C1)*(s1s+s2s+C2))).mean().item()
    wt = torch.tensor([0.0448,0.2856,0.3001,0.2363,0.1333],device=sr_y.device)
    mcs_list,lv,vl = [],[],0
    i1,i2 = sr_y,hr_y
    for _ in range(5):
        if min(i1.shape[2],i1.shape[3])<11: break
        mu1,mu2 = F.conv2d(i1,k,padding=0),F.conv2d(i2,k,padding=0)
        mu1s,mu2s,mu12 = mu1**2,mu2**2,mu1*mu2
        v1=F.conv2d(i1**2,k,padding=0)-mu1s; v2=F.conv2d(i2**2,k,padding=0)-mu2s
        v12=F.conv2d(i1*i2,k,padding=0)-mu12
        mcs_list.append(((2*v12+C2)/(v1+v2+C2)).mean())
        lv=((2*mu12+C1)/(mu1s+mu2s+C1)).mean()
        vl+=1; i1=F.avg_pool2d(i1,2,2); i2=F.avg_pool2d(i2,2,2)
    if vl==0: msssim=0.0
    else:
        w=wt[:vl]/wt[:vl].sum(); mt=torch.stack(mcs_list)
        msssim=(torch.prod(torch.clamp(mt[:-1],min=0)**w[:-1])*torch.clamp(lv,min=0)**w[-1]*torch.clamp(mt[-1],min=0)**w[-1]).item()
    return float(psnr),float(ssim),float(msssim)

KAGGLE_MODEL_DIR = '/kaggle/input/models/qwertywell/model3/pytorch/default/1'

def find_checkpoint(save_dir):
    local = os.path.join(save_dir, 'latest_checkpoint.pth')
    if os.path.exists(local): return local
    kaggle_latest = os.path.join(KAGGLE_MODEL_DIR, 'latest_checkpoint.pth')
    if os.path.exists(kaggle_latest): return kaggle_latest
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--repeat', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size_lr', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--t_max', type=int, default=200000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    accelerator = Accelerator(mixed_precision='fp16')
    set_seed(42 + accelerator.process_index)

    thr, tlr, vhr, vlr = get_df2k_paths()
    if thr is None or not os.path.exists(thr):
        raise FileNotFoundError(f'DF2K not found: {thr}')

    train_ds = DF2KPairedDataset(thr, tlr, patch_size_lr=args.patch_size_lr, scale=4, is_train=True, repeat=args.repeat)
    valid_ds = DF2KPairedDataset(vhr, vlr, scale=4, is_train=False, repeat=1)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    valid_dl = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=2)

    model = NAFNetSR(in_ch=3, out_ch=3, width=64, num_blks=32, upscale=4)
    criterion = nn.L1Loss()
    save_dir = '/kaggle/working/checkpoints'

    start_epoch, best_psnr, best_ssim, best_msssim, global_step = 1, -1.0, -1.0, -1.0, 0

    if args.resume:
        ckpt_path = find_checkpoint(save_dir)
        if ckpt_path is not None:
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                model.load_state_dict(ckpt['model_state_dict'])
                start_epoch = int(ckpt.get('epoch', 0)) + 1
                best_psnr = float(ckpt.get('best_psnr', -1.0))
                best_ssim = float(ckpt.get('best_ssim', -1.0))
                best_msssim = float(ckpt.get('best_msssim', -1.0))
                global_step = int(ckpt.get('global_step', 0))
                if accelerator.is_main_process:
                    print(f'Resumed from: {ckpt_path}')
                    print(f'  Epoch {start_epoch} | Best PSNR: {best_psnr:.3f} dB | Step: {global_step}')
            except Exception as e:
                if accelerator.is_main_process:
                    print(f'Could not resume: {e}')
        else:
            if accelerator.is_main_process:
                print('No checkpoint found, starting fresh.')

    total_epochs = start_epoch + args.epochs - 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.9), weight_decay=0.0)

    if global_step > 0:
        for group in optimizer.param_groups:
            group['initial_lr'] = args.lr

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.t_max, eta_min=1e-7,
        last_epoch=global_step - 1 if global_step > 0 else -1
    )

    model, optimizer, train_dl, valid_dl, scheduler = accelerator.prepare(
        model, optimizer, train_dl, valid_dl, scheduler
    )

    np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        print(f'\\nNAFNet-SR on {accelerator.num_processes} GPUs | Params: {np_:,}')
        print(f'L1 Loss | AdamW betas=(0.9,0.9) lr={args.lr} | Patch {args.patch_size_lr}x{args.patch_size_lr}')
        print(f'Batch/GPU: {args.batch_size} | Epochs: {start_epoch}->{total_epochs} | Steps/Epoch: {len(train_dl)}')

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_dl, desc=f'Epoch [{epoch}/{total_epochs}]', dynamic_ncols=True) if accelerator.is_main_process else train_dl

        for lr_img, hr_img in pbar:
            optimizer.zero_grad()
            loss = criterion(model(lr_img), hr_img)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running_loss += loss.item()
            if accelerator.is_main_process:
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'LR': f'{scheduler.get_last_lr()[0]:.2e}'})

        avg_loss = running_loss / len(train_dl)

        if accelerator.is_main_process:
            eval_model = accelerator.unwrap_model(model)
            eval_model.eval()
            psnr_list, ssim_list, msssim_list = [], [], []
            with torch.no_grad():
                for vl, vh in tqdm(valid_dl, desc='Validating', leave=False, dynamic_ncols=True):
                    vl, vh = vl.to(accelerator.device), vh.to(accelerator.device)
                    p, s, ms = calculate_metrics_gpu(eval_model(vl), vh, scale=4)
                    psnr_list.append(p); ssim_list.append(s); msssim_list.append(ms)

            mp = float(np.mean(psnr_list))
            ms_ = float(np.mean(ssim_list))
            mms = float(np.mean(msssim_list))

            print(f'\\nEpoch [{epoch}/{total_epochs}] Summary:')
            print(f'   - Train Loss     : {avg_loss:.5f}')
            print(f'   - Valid PSNR (Y) : {mp:.3f} dB')
            print(f'   - Valid SSIM (Y) : {ms_:.4f}')

            uw = accelerator.unwrap_model(model)
            if mp > best_psnr:
                best_psnr = mp
                torch.save(uw.state_dict(), os.path.join(save_dir, 'best_psnr_model.pth'))
                print(f'   New Best PSNR Model Saved ({best_psnr:.3f} dB)')
            if ms_ > best_ssim: best_ssim = ms_
            if mms > best_msssim: best_msssim = mms

            torch.save({
                'epoch': int(epoch),
                'model_state_dict': uw.state_dict(),
                'best_psnr': float(best_psnr),
                'best_ssim': float(best_ssim),
                'best_msssim': float(best_msssim),
                'global_step': int(global_step),
            }, os.path.join(save_dir, 'latest_checkpoint.pth'))

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print(f'\\nTraining Complete! Best PSNR: {best_psnr:.3f} dB')

if __name__ == '__main__':
    main()
