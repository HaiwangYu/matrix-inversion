# Matrix-Inversion

```
nvcc matinv_cublas.cu  -lcublas  -o matinv_cublas
```

```
g++ -fPIC -std=c++11 -Wall \
`${ROOTSYS}/bin/root-config --cflags` \
matinv_root.cxx -o matinv_root \
`${ROOTSYS}/bin/root-config --libs` -lMinuit2 -ltbb
```
