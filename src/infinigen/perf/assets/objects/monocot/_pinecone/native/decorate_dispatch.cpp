// SPDX-License-Identifier: BSD-3-Clause
extern "C" void pinecone_extension_scalar(float*,int,const float*);
#ifdef PINECONE_HAVE_AVX2
extern "C" void pinecone_extension_avx2(float*,int,const float*);
#endif
extern "C" void pinecone_extension(float* p,int n,const float* params){
 using Function=void(*)(float*,int,const float*);
#ifdef PINECONE_HAVE_AVX2
 static Function fn=__builtin_cpu_supports("avx2")?pinecone_extension_avx2:pinecone_extension_scalar;
#else
 static Function fn=pinecone_extension_scalar;
#endif
 fn(p,n,params);
}
