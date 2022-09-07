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

```
g++ -fPIC -std=c++11 -Wall \
`${ROOTSYS}/bin/root-config --cflags` \
-I /home/yuhw/sw/eigen-3.4.0/ \
hadamard_product.cxx -o hadamard_product \
`${ROOTSYS}/bin/root-config --libs` -lMinuit2 -ltbb
```

```
nvcc -std=c++11 -m64 -I/home/yuhw/sw/root/include -I /home/yuhw/sw/eigen-3.4.0/ hadamard_product.cu -o hadamard_product \
-L/home/yuhw/sw/root/lib -lCore -lImt -lRIO -lNet -lHist -lGraf -lGraf3d -lGpad -lROOTVecOps -lTree -lTreePlayer -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lThread -lMultiProc -lROOTDataFrame -lm -ldl -lMinuit2 -ltbb
```