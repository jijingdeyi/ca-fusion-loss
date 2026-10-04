Place trained weights in this folder.

The paper default (U-fuser, lambda_halo=1, lambda_washout=0.5, q=0.90) is released as `ours-best.pth`.

```bash
python test_robust.py --checkpoint model/ours-best.pth --data-path /path/to/MSRS/test --outdir results/MSRS/ours
```
