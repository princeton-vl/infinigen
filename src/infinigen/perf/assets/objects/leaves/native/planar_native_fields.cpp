// Field formulas of the v2, ginko and maple leaves (kind 0, 1, 2).
#include "extended_ops.hpp"
#include <cstddef>
using namespace fields;
// These formulas come from Geometry Nodes, so they use its separately rounded
// map and curve rather than the Cycles ones.
#define map map_separate
#define curve curve_separate
namespace {
float valid(float shape){return map(sign(shape),-1,1,1,0);}
float stem(V p,float shift,float len,Curve c,float &raw){
 p.y+=shift;raw=std::fabs(map(curve(c,map(p.y,-1,0)),0,1,-1,1)+p.x);
 float a=raw-smoothmap(p.y,-1.72f,-.35f,.03f,.008f),b=std::fabs(p.y+len)-len;
 return -smoothmin(-a,-b,.02f)-.005f;
}
void v2(V p,const float*a,Curve*c,float*o){
 float xm=map(curve(c[0],map(p.y,-.6f,.6f)),0,1,-1,1)-p.x;
 float w=map(p.y,-70,a[0],a[1],0),ridge=w-std::fabs(xm);
 float mid=map(clamp(divide(w,smoothmin(ridge,map(std::fabs(p.y),0,a[2],1,0),.06f))),.001f,.03f,1,0);
 volatile float projected=map(p.y,-1,1)*(map(curve(c[1],clamp(std::fabs(xm))),0,.9f,0,1.9f)*a[4]);
 float vc=sign(xm)*a[3]+(projected-p.y);
 float dy=p.y-clamp(p.y,-.6f,.6f),shape=std::sqrt(xm*xm+dy*dy)-curve(c[2],map(p.y,-.6f,.6f));
 float veins=mid*map(map(shape,-.3f,0,.015f,0)*map(voronoi1(vc*a[5],.2f),.001f,.05f,1,0),.001f,.01f,1,0);
 float jagged=clamp(voronoi1(vc*a[6])*(a[7]*.05f)+shape);
 o[0]=std::max(map(mid,0,1,1,0),map(jagged,.001f,.002f,1,0))>=.5f;
 o[1]=veins*.005f;o[2]=veins;o[3]=xm;
}
void ginkovein(V p,const float*a,float &vein,float &swell){
 p=p-V{-.18f,0,0};float noise=noise3(p*5),r=length(p),t=pingpong(radial(p));
 float coord=t+r*((t-.5f)*-.44f);
 float fine=wave(noise*.005f+coord,a[3],.6f,3,5,1,-4.62f);
 vein=map(fine*r,.15f,-.32f,0,-.02f);
 swell=wave(noise*.03f+coord,a[4],-.42f,10,1,1,-4.62f)*r;
}
void ginko(V p,const float*a,Curve*c,float*o){
 V q=rotatez(p,a[1]);float angle=radial(q),t=pingpong(angle)*a[2];
 float margin=wave(t+noise1(angle*5)*.3f,a[5],5.82f,1.52f,1,1,0)*.02f;
 float shape=length(V{q.x*.9f,q.y,0})-(margin+curve(c[1],t));
 float raw,st=stem(p,.03f,a[0],c[0],raw),vein,swell;
 o[0]=smoothmin(shape,st,.1f)<=0;if(!o[0])return;
 ginkovein(q,a,vein,swell);
 vein=clamp(clamp((valid(shape)*vein)*map(shape,-1,0,-5,0,false)),0,.01f);
 float z=(valid(st)*raw+vein)*.5f;
 q.z=z;ginkovein(q,a,raw,swell); // Wave is evaluated after the first displacement.
 o[1]=z+map(valid(shape)*swell,0,1,0,.04f);
 o[2]=vein;o[3]=shape;o[4]=p.x;
}
float maple_midrib(V p,float angle,float va,float start,float anneal,float phase,Curve*c){
 V q=rotatez(p,angle);float xm=q.x+curve(c[3],map(q.y))*.1f-.05f;
 V r=rotatez(V{std::fabs(xm),q.y,q.z},va);
 float bend=curve(c[4],map(std::fabs(xm),0,.3f))*.2f,added=r.x+bend;
 float veins=0;
 if(std::fabs(xm)<q.y*anneal&&added<start){
  float vc=(added+sign(xm)*.1f)+phase;
  float width=clamp(.08f-clamp(.05f*length(r)));
  veins=map(voronoi1(vc*8,.7125f),0,width,1,0);
 }
 float ridge=0;
 if(q.y>0)ridge=map(std::fabs(xm)+(noise3(q*10)-.5f)*.01f,0,.01f,1,0);
 return std::max(veins,ridge);
}
void maple(V p,const float*a,Curve*c,float*o){
 V rotated=rotatez(p,a[0]),q=rotatez(rotated,-1.5708f);
 float t=pingpong(radial(q))*a[1],r=length(V{q.x*.9f,q.y,0});
 float shape=(r-curve(c[1],t))-.06f,displacement=(r-curve(c[2],t))-.06f,raw;
 float st=stem(p,.08f,.32f,c[0],raw);
 o[0]=std::min(st,shape)<=0;if(!o[0])return;
 float vein=0;const float angles[]={1.693f,-1.7279f,.8901f,-.9041f,0,3.1416f};
 const float starts[]={-.12f,-.12f,0,0,-.12f,.02f};
 for(int k=0;k<6;k++)vein=std::max(vein,maple_midrib(rotated,angles[k],k==5?.761f:.7854f,starts[k],k==5?10.f:.4f,a[2+k],c));
 V sub={std::fabs(p.x),p.y,0},color=noisecolor(sub*5);
 sub=color*(1-.9f)+sub*.9f;
 float edge=map(voronoi3(sub*150,true),0,.1f),coarse=map(voronoi3(sub*30),0,.1f,0,2,false);
 float subvalue=-(coarse+edge),subcolor=map(edge,0,1,0,-1),area=valid(shape);
 o[2]=area*(1-std::max(vein,map(subcolor,0,-.94f,1,0)));
 float z=(-(std::max(vein,subvalue*-.03f)*.015f))*area+valid(st)*(raw-.01f);
 o[1]=z*.5f+map(length(p)*displacement,-1,0,-.1f,.1f,false);o[3]=rotated.x;
}
}
// Conservative per-asset coverage envelopes, without the vein and contour
// textures; planar_leaf_fields still decides the final silhouette.
extern "C" void planar_leaf_coverage(const float*x,std::size_t w,const float*y,std::size_t h,int kind,const float*a,const float*tables,int size,const float*ranges,float reach,unsigned char*out){
 const int nc=kind==0?5:(kind==1?4:7);Curve c[7];
 for(int i=0;i<nc;i++)c[i]={tables+i*size,size,ranges[2*i],ranges[2*i+1]};
 for(std::size_t row=0;row<h;row++)for(std::size_t col=0;col<w;col++){
  V p={x[col],y[row],0};bool covered;
  if(kind==0){
   float xm=map(curve(c[0],map(p.y,-.6f,.6f)),0,1,-1,1)-p.x;
   float width=map(p.y,-70,a[0],a[1],0),ridge=width-std::fabs(xm);
   float mid=map(clamp(divide(width,smoothmin(ridge,map(std::fabs(p.y),0,a[2],1,0),.06f))),.001f,.03f,1,0);
   float dy=p.y-clamp(p.y,-.6f,.6f),shape=std::sqrt(xm*xm+dy*dy)-curve(c[2],map(p.y,-.6f,.6f));
   // A nonnegative jigsaw displacement can only shrink the blade; a negative
   // amplitude disables culling.
   covered=a[7]<0 || map(mid,0,1,1,0)>=.5f || shape<=.00201f+reach;
  }else if(kind==1){
   V q=rotatez(p,a[1]);float t=pingpong(radial(q))*a[2];
   // Wave output lies in [0,1], so margin <= .02. smoothmin(k=.1)
   // can extend min(shape,stem) by at most k/6; include rounding margin.
   float envelope=length(V{q.x*.9f,q.y,0})-(.02f+curve(c[1],t));
   float raw,st=stem(p,.03f,a[0],c[0],raw);
   covered=std::min(envelope,st)<=.01668f+reach;
  }else{
   V rotated=rotatez(p,a[0]),q=rotatez(rotated,-1.5708f);
   float t=pingpong(radial(q))*a[1],r=length(V{q.x*.9f,q.y,0});
   float shape=(r-curve(c[1],t))-.06f,raw,st=stem(p,.08f,.32f,c[0],raw);
   covered=std::min(st,shape)<=.00001f+reach;
  }
  out[row*w+col]=covered;
 }
}
extern "C" void planar_leaf_fields(const float*xy,std::size_t n,int kind,const float*a,const float*tables,int size,const float*ranges,float*out,int defer,float*bounds,const unsigned char*coverage){
 const int nc=kind==0?5:(kind==1?4:7);Curve c[7];for(int i=0;i<nc;i++)c[i]={tables+i*size,size,ranges[2*i],ranges[2*i+1]};
 float miny=FLT_MAX,maxy=-FLT_MAX,minx=FLT_MAX,maxx=-FLT_MAX,maxz=-FLT_MAX;
 for(std::size_t i=0;i<n;i++){
  float*o=out+6*i;for(int j=0;j<6;j++)o[j]=0;V p={xy[2*i],xy[2*i+1],0};
  if(coverage&&!coverage[i])continue;
  if(kind==0)v2(p,a,c,o);else if(kind==1)ginko(p,a,c,o);else maple(p,a,c,o);
  if(o[0]){miny=std::min(miny,p.y);maxy=std::max(maxy,p.y);maxz=std::max(maxz,o[1]);}
 }
 // Reduce over the same domain as the default before applying each wave. Maple
 // moves to its origin first, which also changes its X-modulated field.
 for(std::size_t i=0;i<n;i++)if(out[6*i]){
  float*o=out+6*i;float xm=kind==1?o[4]:o[3];
  if(kind==2)xm=rotatez(V{xy[2*i],xy[2*i+1]-miny,o[1]-maxz},a[0]).x;
  o[5]=xm;minx=std::min(minx,xm);maxx=std::max(maxx,xm);
 }
 bounds[0]=miny;bounds[1]=maxy;bounds[2]=minx;bounds[3]=maxx;bounds[4]=maxz;
 if(defer)return;
 const int yc=nc-2,xc=nc-1;float sy=kind==0?1.5f:1.f,sx=kind==0?.15f:(kind==1?0:.5f);
 for(std::size_t i=0;i<n;i++)if(out[6*i]){
  float*o=out+6*i;float y=xy[2*i+1],z=o[1];if(kind==2)z-=maxz;
  z+=map(curve(c[yc],map(y,miny,maxy)),0,1,-1,1)*sy;
  z+=map(curve(c[xc],map(o[5],minx,maxx)),0,1,-1,1)*sx;
  o[1]=z;o[5]=y-miny;
 }
}
// The per-asset wave, evaluated on the final mesh vertices. The bounds are the
// ones planar_leaf_fields reduced over the dense raster, not the mesh extrema.
extern "C" void planar_leaf_bend(const float*xyz,std::size_t n,int kind,const float*a,const float*tables,int size,const float*ranges,const float*b,float*out){
 const int nc=kind==0?5:(kind==1?4:7);Curve c[7];
 for(int i=0;i<nc;i++)c[i]={tables+i*size,size,ranges[2*i],ranges[2*i+1]};
 const float sy=kind==0?1.5f:1.f,sx=kind==0?.15f:(kind==1?0:.5f);
 for(std::size_t i=0;i<n;i++){
  V p={xyz[3*i],xyz[3*i+1],xyz[3*i+2]};float xm=p.x,z=p.z;
  if(kind==0)xm=map(curve(c[0],map(p.y,-.6f,.6f)),0,1,-1,1)-p.x;
  if(kind==2){xm=rotatez(V{p.x,p.y-b[0],p.z-b[4]},a[0]).x;z-=b[4];}
  z+=map(curve(c[nc-2],map(p.y,b[0],b[1])),0,1,-1,1)*sy;
  z+=map(curve(c[nc-1],map(xm,b[2],b[3])),0,1,-1,1)*sx;
  out[3*i]=p.x;out[3*i+1]=p.y-b[0];out[3*i+2]=z;
 }
}
