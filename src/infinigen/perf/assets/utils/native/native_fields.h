#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
void broadleaf_fields(const float*,const unsigned char*,size_t,const float*,const float*,int,const float*,float*);
void native_ops(const float*,size_t,int,float*);
void planar_leaf_fields(const float*,size_t,int,const float*,const float*,int,const float*,float*,int,float*,const unsigned char*);
void planar_leaf_coverage(const float*,size_t,const float*,size_t,int,const float*,const float*,int,const float*,float,unsigned char*);
void planar_leaf_bend(const float*,size_t,int,const float*,const float*,int,const float*,const float*,float*);
void bark_displacement(const float*,const float*,const float*,size_t,const float*,float*);
#ifdef __cplusplus
}
#endif
