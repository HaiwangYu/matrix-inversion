#include "TFile.h"
#include "TMatrixD.h"

#include <iostream>


int main(int argc, char *argv[])
{
    int nrep = 1;

    if (argc > 1) {
        nrep = atoi(argv[1]);
    }

    auto *fin = TFile::Open("mat.root", "read");
    TMatrixD *m = (TMatrixD*)fin->Get("mat");

    auto time_start = chrono::high_resolution_clock::now();
    for (int irep=0; irep<nrep; ++irep) {
        m->Invert();
    }
    auto time_stop = chrono::high_resolution_clock::now();
    auto time_duration = chrono::duration_cast<chrono::milliseconds>(time_stop - time_start);
    std::cout<<" ---> time spent/100 calculation: "<<time_duration.count()<<std::endl;
    fin->Close();
    return 0;
}
