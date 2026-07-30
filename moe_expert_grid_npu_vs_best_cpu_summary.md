# NPU vs Best Observed CPU KML Thread Allocation

- CPU candidates: full-grid `8x24`, full-grid `8x32`, full-grid `8x40`, plus representative config-search winners where available.
- KML lib: `third_party/hpckit26/kml26_full/gcclib/sve/kblas/multi/libkblas.so.1.26.0.RC1`

## Best CPU median ms

| experts \ rows/expert | 512 | 256 | 128 | 64 | 32 | 16 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 5.027 | 2.915 | 2.202 | 1.843 | 1.253 | 1.116 | 1.031 |
| 16 | 9.792 | 5.533 | 3.847 | 2.980 | 2.165 | 1.686 | 1.527 |
| 32 | 19.609 | 10.913 | 7.549 | 5.503 | 3.863 | 2.890 | 2.696 |
| 64 | 38.904 | 22.442 | 15.442 | 10.966 | 7.915 | 5.799 | 4.705 |
| 128 | 77.549 | 53.542 | 30.825 | 21.740 | 15.861 | 11.991 | 9.976 |

## Speedup: best CPU ms / NPU ms

| experts \ rows/expert | 512 | 256 | 128 | 64 | 32 | 16 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 20.4x | 12.3x | 9.2x | 7.8x | 5.4x | 4.7x | 4.3x |
| 16 | 21.1x | 20.8x | 16.1x | 12.4x | 9.0x | 6.7x | 6.3x |
| 32 | 22.5x | 20.6x | 25.3x | 20.2x | 14.4x | 10.9x | 10.0x |
| 64 | 22.6x | 22.7x | 28.6x | 22.0x | 16.5x | 12.3x | 10.1x |
| 128 | 22.5x | 27.7x | 26.2x | 21.3x | 16.4x | 12.7x | 10.7x |

## Best CPU config

| experts \ rows/expert | 512 | 256 | 128 | 64 | 32 | 16 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8x32 | 8x32 | search-8x16 | 8x32 | search-8x32 | 8x24 | 8x24 |
| 16 | 8x32 | 8x32 | 8x32 | 8x32 | 8x32 | 8x24 | 8x24 |
| 32 | 8x32 | 8x32 | 8x32 | 8x24 | 8x32 | 8x24 | 8x24 |
| 64 | 8x32 | 8x32 | 8x32 | 8x24 | 8x24 | 8x24 | 8x24 |
| 128 | 8x32 | 8x24 | 8x32 | 8x24 | 8x24 | 8x24 | search-8x24 |

