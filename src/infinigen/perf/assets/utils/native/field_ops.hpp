// Scalar field operations.
// Hash and Voronoi adapted from Blender 4.2 Cycles (Apache-2.0).
// Perlin adapted from Blender 4.2 Cycles / OSL (BSD-3-Clause).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cfloat>

namespace fields {
struct V { float x,y,z; };
inline V operator+(V a,V b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
inline V operator-(V a,V b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
inline V operator*(V a,float s){return {a.x*s,a.y*s,a.z*s};}
inline float dot(V a,V b){return a.x*b.x+a.y*b.y+a.z*b.z;}
inline float length(V a){return std::sqrt(dot(a,a));}
inline float clamp(float a,float lo=0,float hi=1){return std::min(std::max(a,lo),hi);}
inline float divide(float a,float b){return b==0?0:a/b;}
inline float map(float v,float a=0,float b=1,float c=0,float d=1,bool clip=true){
  // Match Cycles' fused multiply/add independently of the host's compiler ISA.
  float r=a==b?0:std::fma(divide(v-a,b-a),d-c,c);
  return clip?clamp(r,std::min(c,d),std::max(c,d)):r;
}
inline float smoothmin(float a,float b,float k){
  if(k==0)return std::min(a,b);
  float h=std::max(k-std::fabs(a-b),0.0f)/k;
  return std::min(a,b)-h*h*h*k*(1.0f/6.0f);
}
inline uint32_t rot(uint32_t v,int n){return (v<<n)|(v>>(32-n));}
inline uint32_t final(uint32_t a,uint32_t b,uint32_t c){
 c^=b;c-=rot(b,14);a^=c;a-=rot(c,11);b^=a;b-=rot(a,25);
 c^=b;c-=rot(b,16);a^=c;a-=rot(c,4);b^=a;b-=rot(a,14);c^=b;c-=rot(b,24);return c;
}
inline uint32_t hash1(uint32_t x){uint32_t a=0xdeadbeef+4+13;return final(a+x,a,a);}
inline uint32_t hash3(uint32_t x,uint32_t y,uint32_t z){uint32_t a=0xdeadbeef+12+13;return final(a+x,a+y,a+z);}
inline uint32_t hash4(uint32_t x,uint32_t y,uint32_t z,uint32_t w){
 uint32_t a=0xdeadbeef+16+13,b=a,c=a;a+=x;b+=y;c+=z;
 a-=c;a^=rot(c,4);c+=b;b-=a;b^=rot(a,6);a+=c;c-=b;c^=rot(b,8);b+=a;
 a-=c;a^=rot(c,16);c+=b;b-=a;b^=rot(a,19);a+=c;c-=b;c^=rot(b,4);b+=a;
 return final(a+w,b,c);
}
inline uint32_t bits(float v){uint32_t u;std::memcpy(&u,&v,4);return u;}
inline float unit(uint32_t v){return float(v)*(1.0f/float(0xffffffffu));}
inline V random3(V p){
 uint32_t x=bits(p.x),y=bits(p.y),z=bits(p.z);
 uint32_t a=0xdeadbeef+16+13,b=a,c=a;a+=x;b+=y;c+=z;
 a-=c;a^=rot(c,4);c+=b;b-=a;b^=rot(a,6);a+=c;c-=b;c^=rot(b,8);b+=a;
 a-=c;a^=rot(c,16);c+=b;b-=a;b^=rot(a,19);a+=c;c-=b;c^=rot(b,4);b+=a;
 // Blender's two hash4 calls share this prefix; computing it once gives the
 // same result.
 return {unit(hash3(x,y,z)),unit(final(a+bits(1.0f),b,c)),unit(final(a+bits(2.0f),b,c))};
}
inline float fade(float t){return t*t*t*(t*(t*6-15)+10);}
inline float grad(uint32_t hash,float x,float y,float z){
 int h=hash&15;float u=h<8?x:y,v=h<4?y:((h==12||h==14)?x:z);
 return ((h&1)?-u:u)+((h&2)?-v:v);
}
inline float perlin(V p){
 int X=int(std::floor(p.x)),Y=int(std::floor(p.y)),Z=int(std::floor(p.z));
 float x=p.x-X,y=p.y-Y,z=p.z-Z,u=fade(x),v=fade(y),w=fade(z),g[8];
 // On a lattice plane the other four corners have zero weight. A zero result
 // still takes the full sum, to keep Cycles' signed zero.
 if(z==0){
  for(int j=0;j<2;j++)for(int i=0;i<2;i++)
   g[j*2+i]=grad(hash3(X+i,Y+j,Z),x-i,y-j,z);
  float u1=1-u,v1=1-v;
  float lower=v1*(g[0]*u1+g[1]*u)+v*(g[2]*u1+g[3]*u);
  if(lower!=0)return lower;
 }
 for(int k=0;k<2;k++)for(int j=0;j<2;j++)for(int i=0;i<2;i++)
   g[k*4+j*2+i]=grad(hash3(X+i,Y+j,Z+k),x-i,y-j,z-k);
 float u1=1-u,v1=1-v,w1=1-w;
 return w1*(v1*(g[0]*u1+g[1]*u)+v*(g[2]*u1+g[3]*u))+
         w*(v1*(g[4]*u1+g[5]*u)+v*(g[6]*u1+g[7]*u));
}
inline float wrap_noise(float p){
 // Cycles' addition, kept for its signed-zero behavior.
 return (std::fabs(p)<100000.f?p:std::fmod(p,100000.f))+.5f*float(std::fabs(p)>=1000000);
}
inline float signed_noise(V p){
 p={wrap_noise(p.x),wrap_noise(p.y),wrap_noise(p.z)};
 return .982f*perlin(p);
}
inline float noise3(V p,float detail=2,float rough=.5f,float lacunarity=2){
 float scale=1,amp=1,maxamp=0,sum=0;
 for(int i=0;i<=int(detail);i++){sum+=signed_noise(p*scale)*amp;maxamp+=amp;amp*=rough;scale*=lacunarity;}
 float r=detail-std::floor(detail),a=.5f*sum/maxamp+.5f;
 if(r==0)return a;
 float b=.5f*(sum+signed_noise(p*scale)*amp)/(maxamp+amp)+.5f;
 return (1-r)*a+r*b;
}
inline float voronoi1(float p,float randomness=1){
 float cell=std::floor(p),local=p-cell,best=FLT_MAX;
 for(int i=-1;i<=1;i++){float q=float(i)+unit(hash1(bits(cell+float(i))))*randomness;best=std::min(best,std::fabs(q-local));}
 return best;
}
inline float voronoi3(V p,bool edge=false,float randomness=1){
 V cell={std::floor(p.x),std::floor(p.y),std::floor(p.z)},local=p-cell,closest={0,0,0},vectors[27];
 float best=FLT_MAX;int n=0;
 for(int k=-1;k<=1;k++)for(int j=-1;j<=1;j++)for(int i=-1;i<=1;i++){
   V off={float(i),float(j),float(k)},q=off+random3(cell+off)*randomness-local;vectors[n++]=q;
   float d=dot(q,q);if(d<best){best=d;closest=q;}
 }
 if(!edge)return std::sqrt(best);
 best=FLT_MAX;
 for(V q:vectors){V d=q-closest;float sq=dot(d,d);if(sq>.0001f)best=std::min(best,dot((closest+q)*.5f,d*(1.0f/std::sqrt(sq))));}
 return best;
}
// Same scalar LUT interpolation/extrapolation order as Cycles float_ramp_lookup.
struct Curve { const float *table; int size; float lo,hi; };
inline float curve(Curve c,float value){
 float f=divide(value-c.lo,c.hi-c.lo),a;
 if(f<0){a=c.table[0]+(c.table[0]-c.table[1])*(-f)*(c.size-1);}
 else if(f>1){a=c.table[c.size-1]+(c.table[c.size-1]-c.table[c.size-2])*(f-1)*(c.size-1);}
 // Operand order is significant: the second weighted term rounds before FMA.
 else {f=clamp(f)*(c.size-1);int i=int(f);float t=f-i;a=c.table[i];if(t>0)a=std::fma(1-t,a,t*c.table[i+1]);}
 return clamp(a);
}
}
