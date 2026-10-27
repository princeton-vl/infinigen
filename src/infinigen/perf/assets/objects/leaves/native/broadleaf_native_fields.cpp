// Broadleaf field formulas, before the wave displacement.
#include "field_ops.hpp"
#include <cstddef>
using namespace fields;

// Curves: midrib, width, vein0, vein1, vein2, subvein, y profile, shape profile.
struct Params {float midrib_length,stem_length,asymmetry,angle,density,seed,jigsaw_scale,jigsaw_depth,alternate1,alternate2,jigsaw_amplitude;};
inline float midrib(V p,float xm,const Params &a,float noise){
 float width=map(noise,0,1,-1,1)*.01f+map(p.y,-70,a.midrib_length,a.midrib_length,0);
 float ridge=width-std::fabs(xm),stem=map(std::fabs(p.y),0,a.stem_length,1,0);
 return map(clamp(divide(1,smoothmin(ridge,stem,.06f))),.001f,.03f,1,0);
}
inline float coord(float xm,float y,float width,const Params&a,Curve c,bool sub=false){
 float side=a.asymmetry*((xm>0)-(xm<0));
 float t=sub?xm:clamp(divide(clamp(std::fabs(xm)),width));
 // Materialize the multiply node before the subtract node: Cycles evaluates
 // them separately, rather than contracting this whole expression to FMA.
 volatile float projected=map(y,-1,1)*(map(curve(c,t),0,1,0,1.9f)*a.angle);
 return side+(projected-y);
}
inline float random_mask(float c,float shape,float density,float alternate_density){
 float v=voronoi1(c*density,.2f),alt=voronoi1(c*alternate_density);
 float gate=std::floor(alt+.35f+.5f);
 return map(shape*map(v+gate,0,.02f,.95f,0),.001f,.005f,1,0);
}
inline float veins(float noise,float mid,float edge,float m1,float m2,float base){
 float perturbed=base+map(noise,0,1,-1,1)*.02f;
 float shaped=map(edge*map(perturbed,0,.03f,1,0),.001f,.01f,1,0);
 return mid*(m2*(m1*shaped));
}
extern "C" void broadleaf_fields(const float*xy,const unsigned char*mask,std::size_t n,
 const float*params,const float*tables,int table_size,const float*ranges,float*output){
 Params a={params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9],params[10]};
 Curve curves[8];for(int i=0;i<8;i++)curves[i]={tables+i*table_size,table_size,ranges[2*i],ranges[2*i+1]};
 for(std::size_t i=0;i<n;i++){
  float*o=output+8*i;if(mask&&!mask[i]){for(int j=0;j<8;j++)o[j]=0;continue;}
  float x=xy[2*i],y=xy[2*i+1],ymap=map(y,-.6f,.6f);
  float xm=map(curve(curves[0],ymap),0,1,-1,1)-x;
  float width=curve(curves[1],ymap),dy=y-clamp(y,-.6f,.6f);
  float shape=std::sqrt(xm*xm+dy*dy)-width,c[3];
  for(int j=0;j<3;j++)c[j]=coord(xm,y,width,a,curves[2+j]);
  float ew=map(shape,-.3f,.05f,.015f,0);
  float m1=random_mask(c[0],ew,a.density,a.alternate1),m2=random_mask(c[2],ew,a.density,a.alternate2),base=voronoi1(c[1]*a.density,.2f);
  V p={x,y,0};float noise=noise3(p*20),z0=veins(noise,midrib(p,xm,a,noise),ew,m1,m2,base)*.005f;
  p.z=z0;noise=noise3(p*20);float displaced=midrib(p,xm,a,noise),offset=veins(noise,displaced,ew,m1,m2,base);
  float jagged=clamp(voronoi1(c[1]*a.jigsaw_scale)*a.jigsaw_amplitude+shape);
  float coverage=std::max(map(displaced,0,1,1,0),map(jagged,.001f,.002f,1,0));
  V subp={std::fabs(xm),coord(xm,y,width,a,curves[5],true),0};
  float coarse=map(voronoi3(subp*30),0,.1f,0,2,false),edge=map(voronoi3(subp*150,true),0,.1f);
  float sub=-(coarse+edge),color=map(edge,0,1,0,-1);
  float curvature=(curve(curves[6],ymap)*curve(curves[7],map(shape,0,-1)))*.7f;
  float z=(z0+curvature)+sub*.0002f;p.z=z;
  noise=noise3(p*20);float material=veins(noise,midrib(p,xm,a,noise),ew,m1,m2,base);
  o[0]=coverage;o[1]=z;o[2]=x;o[3]=y;o[4]=offset;o[5]=color;o[6]=material;o[7]=xm;
 }
}
extern "C" void native_ops(const float*xyz,std::size_t n,int op,float*out){
 for(std::size_t i=0;i<n;i++){V p={xyz[i*3],xyz[i*3+1],xyz[i*3+2]};
 out[i]=op==0?noise3(p):(op==1?voronoi1(p.x,.2f):(op==2?voronoi3(p):voronoi3(p,true)));}
}
