// SPDX-License-Identifier: BSD-3-Clause
// Scalar pinecone radial noise, using the native field operations.
#ifdef PINECONE_SCALAR_ENTRY
#define pinecone_extension pinecone_extension_scalar
#endif
#include "field_ops.hpp"
extern "C" void pinecone_extension(float* xyz,int count,const float* p){
 for(int i=0;i<count;++i){fields::V v{xyz[3*i],xyz[3*i+1],xyz[3*i+2]};float len=fields::length(v);float inv=len==0?0:1.f/len;
  fields::V dir=(v*inv+fields::V{p[2],p[3],p[4]})*p[1];
  float noise=fields::signed_noise(dir)+.25f*fields::signed_noise(dir*2.f);
  float strength=(noise+.25f)*p[0];v=v+v*strength;
  xyz[3*i]=v.x;xyz[3*i+1]=v.y;xyz[3*i+2]=v.z;
 }
}
