# Zeisel (z=2, 10, 20):
python scRAE.py --model scRAE --batch_size 16 --n_l 1 --g_h_l1 1024 --g_h_l2 512 --g_h_l3 512 --g_h_l4 512 --lg_h_l1 32  --lg_h_l2 32 --lg_h_l3 32 --lg_h_l4 32 --d_h_l1 64 --d_h_l2 64 --d_h_l3 64 --d_h_l4 64 --actv sig --trans sparse --bn --leak 0.2 --lam 10.0 --epoch 200 --z_dim 2 --ae_lr 0.0015 --gan_lr 0.0015 --train --dataset Zeisel
python scRAE.py --model scRAE --batch_size 16 --n_l 2 --g_h_l1 512 --g_h_l2 512 --g_h_l3 512 --g_h_l4 512 --lg_h_l1 32  --lg_h_l2 32 --lg_h_l3 32 --lg_h_l4 32 --d_h_l1 64 --d_h_l2 64 --d_h_l3 64 --d_h_l4 64 --actv sig --trans sparse --bn --leak 0.2 --lam 10.0 --epoch 200 --z_dim 10 --ae_lr 0.001 --gan_lr 0.00009 --train --dataset Zeisel
python scRAE.py --model scRAE --batch_size 16 --n_l 1 --g_h_l1 512 --g_h_l2 512 --g_h_l3 512 --g_h_l4 512 --lg_h_l1 32  --lg_h_l2 32 --lg_h_l3 32 --lg_h_l4 32 --d_h_l1 64 --d_h_l2 64 --d_h_l3 64 --d_h_l4 64 --actv sig --trans sparse --bn --leak 0.2 --lam 10.0 --epoch 200 --z_dim 20 --ae_lr 0.001 --gan_lr 0.00009 --train --dataset Zeisel