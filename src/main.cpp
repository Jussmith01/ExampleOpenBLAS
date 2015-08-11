#include <iostream> // Needed for cout
#include <cstring> // Needed for memcpy
#include <vector> // Needed for vector

/* If using openblas must include flag -lopenblas on linker */
#include <lapacke.h> // Contains all lapack types and functions

/* If using openblas must include flag -lopenblas on linker */
#include <cblas.h> // Contains all blas types and functions

int main (int argc,char* argv[])
{
    // Some needed ints. n=size of matrix, info = error checker from ssyev
    lapack_int n=5,info;

    /* Declare matrix of values! */
    float atemp[n*n] = {
        1.96f, -6.49f, -0.47f, -7.20f, -0.65f,
        0.00f,  3.80f, -6.39f,  1.50f, -6.34f,
        0.00f,  0.00f, 4.17f, -1.51f, 2.67f,
        0.00f,  0.00f, 0.00f,  5.70f, 1.80f,
        0.00f,  0.00f, 0.00f,  0.00f, -7.10f
    };

    /* Copy to a vector! */
    std::vector<float> a(n*n);

    /* Declare vector to hold eigenvalues! */
    std::vector<float> w(n);

    /* Move data to the std::vector a*/
    std::memcpy(&a[0],atemp,n*n*sizeof(float));

    /* Make a working vector! */
    std::vector<float> c(n*n);
    std::memcpy(&c[0],&a[0],n*n*sizeof(float));

    /* -------------------- Eigenvalues with lapack ----------------- */

    /* Example of how to use std::vector with ssyev */
    info = LAPACKE_ssyev( LAPACK_ROW_MAJOR,'V','U',n,&c[0],n,&w[0] );

    /* Error checking */
    if (info > 0)
    {
        std::cout << "Diagonalization failed! Error code: " << info << std::endl;
        exit(EXIT_FAILURE);
    }
    /* -------------------------------------------------------------- */

    /* Print the eigenvalues */
    std::cout << "Eigenvalues: " << std::endl;
    for (int j=0;j<n;++j)
    {
            std::cout << w[j] << " ";
    }
    std::cout << std::endl;

    /* Print the eigenvectors */
    std::cout << "Eigenvectors: " << std::endl;
    for (int i=0;i<n;++i)
    {
        for (int j=0;j<n;++j)
        {
            std::cout << c[i*n+j] << " ";
        }
        std::cout << std::endl;
    }

    std::memcpy(&c[0],&a[0],n*n*sizeof(float));

    /* -------------------- Matrix inverse with Lapack ----------------- */
    // Seg faults on my machine, some problem with pthreads, but you get the point
    lapack_int ipiv[n];
    LAPACK_sgetrf(&n,&n,&c[0],&n,ipiv,&info);
    // Error checking
    if (info > 0)
    {
        std::cout << "Inversion failed! Error code: " << info << std::endl;
        exit(EXIT_FAILURE);
    }

    info = LAPACKE_sgetri( n,n,&c[0],n,ipiv );
    // Error checking
    if (info > 0)
    {
        std::cout << "Inversion failed! Error code: " << info << std::endl;
        exit(EXIT_FAILURE);
    }
    /* ----------------------------------------------------------------- */

    /* -------------------- Matrix Mult with cBlas ----------------- */

    std::vector<float> b(n*n);
    cblas_sgemm(CblasRowMajor,CblasTrans,CblasNoTrans,n,n,1,1.0f,&a[0],n,&c[0],n,0.0f,&b[0],n);
    //cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,n,1,1.0f,&c[0],n,&a[0],n,0.0f,&b[0],n);

    /* -------------------------------------------------------------- */

    /* Print the eigenvectors */
    std::cout << "Matrx Mult Result: " << std::endl;
    for (int i=0;i<n;++i)
    {
        for (int j=0;j<n;++j)
        {
            std::cout << b[i*n+j] << " ";
        }
        std::cout << std::endl;
    }


    return 0;
}
