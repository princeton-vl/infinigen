// Field operations used by the planar leaf formulas.
// Wave/noise follow Blender 4.2 Cycles, under the licenses in NOTICE.md.
#pragma once
#include "field_ops.hpp"
namespace fields {
inline float sign(float x){return float((x>0)-(x<0));}
inline float smoothmap(float x,float a,float b,float c,float d){
 float t=clamp(divide(x-a,b-a));return c+(d-c)*(t*t*(3-2*t));
}
inline V rotatez(V p,float angle){float c=std::cos(angle),s=std::sin(angle);return {c*p.x-s*p.y,s*p.x+c*p.y,p.z};}
inline float radial(V p){return std::atan2(p.y,p.x)/(2*3.14159265358979323846f)+.5f;}
inline float pingpong(float a,float scale=.5f){if(scale==0)return 0;float t=(a-scale)/(scale*2);return std::fabs((t-std::floor(t))*scale*2-scale);}
inline float grad1(uint32_t h,float x){float g=1+(h&7);return ((h&8)?-g:g)*x;}
inline float snoise1(float p){
 p=wrap_noise(p);
 int i=int(std::floor(p));float f=p-i,t=fade(f);
 return .25f*((1-t)*grad1(hash1(i),f)+t*grad1(hash1(i+1),f-1));
}
inline float noise1(float p){float sum=0,amp=1,scale=1,total=0;for(int i=0;i<=2;i++){sum+=snoise1(p*scale)*amp;total+=amp;amp*=.5f;scale*=2;}return .5f*sum/total+.5f;}
inline uint32_t hash2(uint32_t x,uint32_t y){uint32_t a=0xdeadbeef+8+13;return final(a+x,a+y,a);}
inline V noiseoffset(float seed){return {100+unit(hash2(bits(seed),bits(0.f)))*100,100+unit(hash2(bits(seed),bits(1.f)))*100,100+unit(hash2(bits(seed),bits(2.f)))*100};}
inline V noisecolor(V p){return {noise3(p),noise3(p+noiseoffset(3)),noise3(p+noiseoffset(4))};}
inline float wave(float x,float scale,float distortion,float detail,float dscale,float rough,float phase){
 V p=V{x*scale,0,0};p=(p+V{.000001f,.000001f,.000001f})*.999999f;
 float n=p.x*20+phase;
 n+=distortion*(noise3(p*dscale,detail,rough)*2-1);
 return .5f+.5f*std::sin(n-1.57079632679489661923f);
}
// Geometry Nodes rounds the multiply and the add separately here, unlike the
// fused Cycles map in field_ops.hpp.
inline float map_separate(float v,float a=0,float b=1,float c=0,float d=1,bool clip=true){
 volatile float product=divide(v-a,b-a)*(d-c);float r=a==b?0:c+product;
 return clip?clamp(r,std::min(c,d),std::max(c,d)):r;
}
inline float curve_separate(Curve c,float value){
 float f=divide(value-c.lo,c.hi-c.lo)*(c.size-1);
 if(f<0)return clamp(c.table[0]+(c.table[0]-c.table[1])*(-f));
 if(f>c.size-1)return clamp(c.table[c.size-1]+(c.table[c.size-1]-c.table[c.size-2])*(f-(c.size-1)));
 int i=int(f);float t=f-i;if(t==0)return clamp(c.table[i]);
 volatile float left=(1-t)*c.table[i],right=t*c.table[i+1];return clamp(left+right);
}
}
