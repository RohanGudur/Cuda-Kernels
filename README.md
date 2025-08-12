# Cuda-Kernels

This repository contains CUDA implementations of various algorithms, mostly related to deep learning.  

I am currently ranked in the **Top 20** on [leetgpu.com](https://leetgpu.com) — Id like to thank these platforms for making GPUs more easily accesible along with the og collab on which I ran nsight compute to rofile my kernels 

There is also a **Colab notebook** in this repo that can help you do the same 
I belive wiritjng kernel sis mostly about making them better as and sugges tyou tink from first pricniples inituition has played a mjoir role for me and nisght compoute either supootes it or ghive you pointers onw whtas off in many ways opitmisation of these kernels take more time than implementing code so spend more time trying to make it better thats where the learning lies 

---

## 📂 Repository Structure & Profiling Notes

Wherever possible, I have included **Nsight Compute (`ncu`) profiling results** inside the kernel subfolders. This is to show how different implementations of the same algorithm can vary in performance.  

Some kernels also include:  
- CUDA events for timing execution  
- The GPU model used for benchmarking  
- Comments explaining specific performance choices  

