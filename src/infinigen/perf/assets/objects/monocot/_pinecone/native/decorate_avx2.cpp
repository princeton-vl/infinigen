// SPDX-License-Identifier: BSD-3-Clause
// AVX2 eight-lane evaluation of the field_ops.hpp pinecone radial noise.
// Hash provenance: Blender 4.2 Cycles (Apache-2.0).
// Perlin provenance: Blender 4.2 Cycles / OSL (BSD-3-Clause).
// Reached through decorate_dispatch; the scalar kernel handles the tail.
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to=function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif
#include <immintrin.h>
#include <cmath>
#include <cstdint>
extern "C" void pinecone_extension_scalar(float*,int,const float*);
namespace {
using V=__m256;using I=__m256i;
inline V c(float x){return _mm256_set1_ps(x);}inline I ci(int x){return _mm256_set1_epi32(x);}
inline V add(V a,V b){return _mm256_add_ps(a,b);}inline V sub(V a,V b){return _mm256_sub_ps(a,b);}inline V mul(V a,V b){return _mm256_mul_ps(a,b);}
template<int N>inline I rot(I x){return _mm256_or_si256(_mm256_slli_epi32(x,N),_mm256_srli_epi32(x,32-N));}
inline I hash(I x,I y,I z){I base=ci(int(0xdeadbeefU+12+13)),a=_mm256_add_epi32(base,x),b=_mm256_add_epi32(base,y),d=_mm256_add_epi32(base,z);
 d=_mm256_sub_epi32(_mm256_xor_si256(d,b),rot<14>(b));a=_mm256_sub_epi32(_mm256_xor_si256(a,d),rot<11>(d));b=_mm256_sub_epi32(_mm256_xor_si256(b,a),rot<25>(a));
 d=_mm256_sub_epi32(_mm256_xor_si256(d,b),rot<16>(b));a=_mm256_sub_epi32(_mm256_xor_si256(a,d),rot<4>(d));b=_mm256_sub_epi32(_mm256_xor_si256(b,a),rot<14>(a));return _mm256_sub_epi32(_mm256_xor_si256(d,b),rot<24>(b));}
inline V select(I mask,V a,V b){return _mm256_blendv_ps(b,a,_mm256_castsi256_ps(mask));}
inline V grad(I h,V x,V y,V z){h=_mm256_and_si256(h,ci(15));V u=select(_mm256_cmpgt_epi32(ci(8),h),x,y);I special=_mm256_or_si256(_mm256_cmpeq_epi32(h,ci(12)),_mm256_cmpeq_epi32(h,ci(14)));V v=select(_mm256_cmpgt_epi32(ci(4),h),y,select(special,x,z));
 I s1=_mm256_slli_epi32(_mm256_and_si256(h,ci(1)),31),s2=_mm256_slli_epi32(_mm256_and_si256(h,ci(2)),30);return add(_mm256_xor_ps(u,_mm256_castsi256_ps(s1)),_mm256_xor_ps(v,_mm256_castsi256_ps(s2)));}
inline V fade(V t){return mul(mul(mul(t,t),t),add(mul(t,sub(mul(t,c(6)),c(15))),c(10)));}
inline V noise(V px,V py,V pz){px=add(px,c(0));py=add(py,c(0));pz=add(pz,c(0));V fx=_mm256_floor_ps(px),fy=_mm256_floor_ps(py),fz=_mm256_floor_ps(pz);I X=_mm256_cvttps_epi32(fx),Y=_mm256_cvttps_epi32(fy),Z=_mm256_cvttps_epi32(fz);V x=sub(px,fx),y=sub(py,fy),z=sub(pz,fz),u=fade(x),v=fade(y),w=fade(z),g[8];
 for(int k=0;k<2;++k)for(int j=0;j<2;++j)for(int i=0;i<2;++i)g[k*4+j*2+i]=grad(hash(_mm256_add_epi32(X,ci(i)),_mm256_add_epi32(Y,ci(j)),_mm256_add_epi32(Z,ci(k))),sub(x,c(i)),sub(y,c(j)),sub(z,c(k)));
 V u1=sub(c(1),u),v1=sub(c(1),v),w1=sub(c(1),w);V lower=add(mul(v1,add(mul(g[0],u1),mul(g[1],u))),mul(v,add(mul(g[2],u1),mul(g[3],u))));V upper=add(mul(v1,add(mul(g[4],u1),mul(g[5],u))),mul(v,add(mul(g[6],u1),mul(g[7],u))));return mul(c(.982f),add(mul(w1,lower),mul(w,upper)));}
}
extern "C" void pinecone_extension_avx2(float*xyz,int count,const float*p){
 // the packet path covers the small frequencies and offsets the asset uses;
 // non-finite parameters, or frequencies and offsets above 100, go to the scalar kernel
 if(!std::isfinite(p[0])||!std::isfinite(p[1])||std::fabs(p[1])>100||!std::isfinite(p[2])||!std::isfinite(p[3])||!std::isfinite(p[4])||std::fabs(p[2])>100||std::fabs(p[3])>100||std::fabs(p[4])>100){pinecone_extension_scalar(xyz,count,p);return;}
 const I offsets=_mm256_setr_epi32(0,3,6,9,12,15,18,21);int i=0;
 for(;i+8<=count;i+=8){float*base=xyz+3*i;V x=_mm256_i32gather_ps(base,offsets,4),y=_mm256_i32gather_ps(base+1,offsets,4),z=_mm256_i32gather_ps(base+2,offsets,4);V len=_mm256_sqrt_ps(add(add(mul(x,x),mul(y,y)),mul(z,z)));V inv=_mm256_div_ps(c(1),len);inv=_mm256_blendv_ps(inv,c(0),_mm256_cmp_ps(len,c(0),_CMP_EQ_OQ));V dx=mul(add(mul(x,inv),c(p[2])),c(p[1])),dy=mul(add(mul(y,inv),c(p[3])),c(p[1])),dz=mul(add(mul(z,inv),c(p[4])),c(p[1]));
  V n=add(noise(dx,dy,dz),mul(c(.25f),noise(mul(dx,c(2)),mul(dy,c(2)),mul(dz,c(2)))));V strength=mul(add(n,c(.25f)),c(p[0]));
  float a[8],b[8],d[8];_mm256_storeu_ps(a,add(x,mul(x,strength)));_mm256_storeu_ps(b,add(y,mul(y,strength)));_mm256_storeu_ps(d,add(z,mul(z,strength)));for(int j=0;j<8;++j){base[3*j]=a[j];base[3*j+1]=b[j];base[3*j+2]=d[j];}
 }
 if(i<count)pinecone_extension_scalar(xyz+3*i,count-i,p);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
