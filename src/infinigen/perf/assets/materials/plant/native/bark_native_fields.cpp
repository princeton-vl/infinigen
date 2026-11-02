// The BarkRandom geometry displacement, evaluated per point.
#include "field_ops.hpp"
#include <cstddef>
using namespace fields;

inline V cross(V a,V b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
inline V normalize(V a){float n=length(a);return n==0?V{0,0,0}:a*(1.0f/n);}
inline V rotate_axis_angle(V point,V center,V axis,float angle){
  V v=point-center;
  float n=length(axis);
  if(n==0 || angle==0)return point;
  V u=axis*(1.0f/n);
  float c=std::cos(angle),s=std::sin(angle);
  return center+v*c+cross(u,v)*s+u*(dot(u,v)*(1-c));
}
inline V canonical(V position,V parent,V skeleton){
  V stem=skeleton-parent,z={0,0,1};
  float stem_length=length(stem);
  float cosine=stem_length==0?0:clamp(dot(stem,z)/stem_length,-1,1);
  V rotated=rotate_axis_angle(position,parent,cross(stem,z),std::acos(cosine));
  return (rotated-V{parent.x,parent.y,0})*2.0f;
}
inline float radius_at(V position,V parent,V skeleton){
  V axis=normalize(skeleton-parent),delta=position-parent;
  V center=parent+axis*dot(axis,delta);
  return length(center-position);
}
inline float edge_voronoi(V p,float scale,float randomness=1){
  return map(voronoi3(p*scale,true,randomness),0,.1f);
}
extern "C" void bark_displacement(
 const float*positions,const float*parents,const float*skeletons,std::size_t n,
 const float*params,float*output){
  const float displacement_scale=params[0];
  const float z_noise_scale=params[1];
  const float z_noise_amount=params[2];
  const float z_multiplier=params[3];
  const float primary_scale=params[4];
  const float primary_randomness=params[5];
  const float secondary_weight=params[6];
  const float secondary_scale=params[7];
  // params[8] (Noise Texture Scale) is a node-group input the displacement ignores.
  const float noise_detail=params[9];
  const float noise_weight=params[10];
  for(std::size_t i=0;i<n;i++){
    V position={positions[3*i],positions[3*i+1],positions[3*i+2]};
    V parent={parents[3*i],parents[3*i+1],parents[3*i+2]};
    V skeleton={skeletons[3*i],skeletons[3*i+1],skeletons[3*i+2]};
    V coordinate=canonical(position,parent,skeleton);
    float secondary=edge_voronoi(coordinate,secondary_scale);
    float z_noise=noise3(coordinate*z_noise_scale);
    V warped={coordinate.x,coordinate.y,(coordinate.z+z_noise*z_noise_amount)*z_multiplier};
    float primary=edge_voronoi(warped,primary_scale,primary_randomness);
    float mixed=secondary*secondary_weight+primary*(1.0f-secondary_weight);
    float noise=noise3(coordinate*noise_weight,noise_detail);
    float field=(mixed+noise*z_noise_scale)*displacement_scale;
    output[i]=radius_at(position,parent,skeleton)*1.5f*field;
  }
}
