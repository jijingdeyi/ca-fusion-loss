Place public IR–VIS datasets here, or set `IVIF_DATA_ROOT` in `.env` to an existing root.

Expected layout:

```
$IVIF_DATA_ROOT/
  MSRS/
    train/{ir,vi}/
    test/{ir,vi}/
    detection/{ir,vi}/          # 80-image YOLO subset
  llvip_test/{ir,vi}/           # 40-pair eval split used in the paper
  RoadScene/test/{ir,vi}/
  FMB/test/{ir,vi}/
  M3FD/M3FD_Fusion/{ir,vi}/
  OpIVF/test/{ir,vi}/
```

Do not commit the images. They are public datasets with their own licenses.
