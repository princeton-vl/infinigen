#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cfloat>

namespace infinigen::heightfield {

struct QuadVec3 { float x, y, z; };

inline QuadVec3 quad_sub(QuadVec3 a, QuadVec3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
inline QuadVec3 quad_cross(QuadVec3 a, QuadVec3 b) {
    volatile float x0=a.y*b.z,x1=a.z*b.y,y0=a.z*b.x,y1=a.x*b.z,z0=a.x*b.y,z1=a.y*b.x;
    return {x0-x1,y0-y1,z0-z1};
}
inline float quad_dot(QuadVec3 a, QuadVec3 b) {
    volatile float x=a.x*b.x,y=a.y*b.y,z=a.z*b.z,xy=x+y;
    return xy+z;
}
inline float projected_length(QuadVec3 p,QuadVec3 q,QuadVec3 n) {
    QuadVec3 d=quad_sub(p,q);float dn=quad_dot(d,n);volatile float dn2=dn*dn;
    return std::sqrt(std::max(quad_dot(d,d)-dn2,0.0f));
}
inline float projected_area(QuadVec3 p,QuadVec3 q,QuadVec3 r,QuadVec3 n) {
    return std::fabs(quad_dot(quad_cross(quad_sub(q,p),quad_sub(r,p)),n));
}
inline void triangulate_quads_beauty(const float* vertices,const std::uint32_t* quads,std::size_t count,std::uint32_t* faces) {
    for(std::size_t i=0;i<count;++i){
        auto point=[&](int j){const float*p=vertices+std::size_t(quads[i*4+j])*3;return QuadVec3{p[0],p[1],p[2]};};
        QuadVec3 a=point(1),b=point(2),c=point(3),d=point(0);
        QuadVec3 n0=quad_cross(quad_sub(c,b),quad_sub(d,b)),n1=quad_cross(quad_sub(d,b),quad_sub(a,b));
        QuadVec3 n={n0.x+n1.x,n0.y+n1.y,n0.z+n1.z};float nl=std::sqrt(quad_dot(n,n));
        if(nl>0){n.x/=nl;n.y/=nl;n.z/=nl;}else n={0,0,1};
        float ab=projected_length(a,b,n),bc=projected_length(b,c,n),cd=projected_length(c,d,n);
        float da=projected_length(d,a,n),bd=projected_length(b,d,n),ac=projected_length(a,c,n);
        volatile float p02a=bc+cd,p02b=da+ab,p13a=ab+bc,p13b=cd+da;
        float q02=projected_area(b,c,d,n)/std::max(float(p02a+bd),FLT_MIN)+projected_area(b,d,a,n)/std::max(float(p02b+bd),FLT_MIN);
        float q13=projected_area(a,b,c,n)/std::max(float(p13a+ac),FLT_MIN)+projected_area(a,c,d,n)/std::max(float(p13b+ac),FLT_MIN);
        const std::uint32_t*q=quads+i*4;std::uint32_t*f=faces+i*6;
        if(q02>q13){f[0]=q[0];f[1]=q[1];f[2]=q[2];f[3]=q[0];f[4]=q[2];f[5]=q[3];}
        else{f[0]=q[1];f[1]=q[3];f[2]=q[0];f[3]=q[3];f[4]=q[1];f[5]=q[2];}
    }
}

}
