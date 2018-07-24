==================================================
LLVM* OpenMP* Runtime with User-defined Scheduling
==================================================
This repository contains changes to enable user-defined scheduling in OpenMP parallel loop. 
Our work has been published in ICPP `19
(https://dl.acm.org/citation.cfm?id=3337913)

=====================================================================
How to Build the LLVM* OpenMP* Libraries with User-defined Scheduling
=====================================================================

- Detailed instructions to build LLVM OpenMP are explained in README_old.rst
- In addition to the settings based on the instructions, LIBOMP_ENABLE_USERSCHED should be turned on to enable user-defined scheduling. 

  .. code-block:: console

    $ mkdir build
    $ cd build
    $ cmake ../ -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release 
    -DLIBOMP_ENABLE_USERSCHED=On
    $ make

- To load the runtime library ahead of the builtin runtime library in gcc, icc and clang, LD_PRELOAD should be used.

  .. code-block:: console 

    $ export LD_PRELOAD=<build_directory>/runtime/src/libomp.so

- To use API we suggested in the paper, the modified omp.h should be included. So, you need to include the directory where the header is. The headers should be included with the following compiler option.
  
  .. code-block:: console
  
    -I<build_directory>/runtime/src

- Among two APIs we suggested in the paper, the subspace select function is not fully supported. The subspace selection is fixed as described in the paper. We'll support it later. 
