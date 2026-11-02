// SPDX-License-Identifier: BSD-3-Clause
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cstring>
#include <cmath>
struct V {double x=0,y=0,z=0; V operator+(V b)const{return{x+b.x,y+b.y,z+b.z};} V operator*(double s)const{return{x*s,y*s,z*s};} V& operator+=(V b){x+=b.x;y+=b.y;z+=b.z;return *this;}};
struct Edge {int a,b,f0=-1,f1=-1;};
struct R {std::vector<V> co;std::vector<int> faces;};
extern "C" void* subdivide(const float* co,int nv,const int* loops,const int* counts,int nf,int limit,int* sizes){
 std::vector<V> p(nv),fp(nf),sumF(nv),sumE(nv);std::vector<int> degree(nv),offset(nf+1),le;
 for(int i=0;i<nv;++i)p[i]={co[3*i],co[3*i+1],co[3*i+2]};
 std::vector<Edge> edges;std::unordered_map<uint64_t,int> map;
 int nl=0;for(int f=0;f<nf;++f){offset[f]=nl;for(int j=0;j<counts[f];++j)fp[f]+=p[loops[nl+j]];fp[f]=fp[f]*(1./counts[f]);nl+=counts[f];}offset[nf]=nl;le.resize(nl);
 for(int f=0;f<nf;++f)for(int j=0;j<counts[f];++j){int a=loops[offset[f]+j],b=loops[offset[f]+(j+1)%counts[f]];int lo=std::min(a,b),hi=std::max(a,b);uint64_t key=(uint64_t(lo)<<32)|uint32_t(hi);auto it=map.find(key);int e;
  if(it==map.end()){e=edges.size();map[key]=e;edges.push_back({lo,hi,f,-1});}else{e=it->second;edges[e].f1=f;}le[offset[f]+j]=e;sumF[a]+=fp[f];degree[a]++;}
 R*r=new R;r->co.resize(nv+edges.size()+nf);r->faces.resize(nl*4);
 std::vector<V> boundary(nv);std::vector<int> bn(nv);
 for(int i=0;i<(int)edges.size();++i){auto e=edges[i];V mid=(p[e.a]+p[e.b])*.5;sumE[e.a]+=mid;sumE[e.b]+=mid;
  if(e.f1<0){r->co[nv+i]=mid;boundary[e.a]+=p[e.b];boundary[e.b]+=p[e.a];bn[e.a]++;bn[e.b]++;}
  else r->co[nv+i]=(p[e.a]+p[e.b]+fp[e.f0]+fp[e.f1])*.25;}
 for(int i=0;i<nv;++i){int n=degree[i];r->co[i]=bn[i]? p[i]*.75+boundary[i]*.125 : (sumF[i]*(1./n)+sumE[i]*(2./n)+p[i]*(n-3))*(1./n);}
 for(int f=0;f<nf;++f){r->co[nv+edges.size()+f]=fp[f];for(int j=0;j<counts[f];++j){int k=offset[f]+j;r->faces[4*k]=loops[k];r->faces[4*k+1]=nv+le[k];r->faces[4*k+2]=nv+edges.size()+f;r->faces[4*k+3]=nv+le[offset[f]+(j+counts[f]-1)%counts[f]];}}
 if(limit){std::vector<V> out(r->co.size()),diag(r->co.size()),neighbors(r->co.size());std::vector<int> n(r->co.size());
  for(int f=0;f<nl;++f)for(int j=0;j<4;++j){int a=r->faces[4*f+j];neighbors[a]+=r->co[r->faces[4*f+(j+1)%4]];diag[a]+=r->co[r->faces[4*f+(j+2)%4]];n[a]++;}
  for(int i=0;i<(int)out.size();++i)out[i]=(r->co[i]*(n[i]*n[i])+neighbors[i]*4+diag[i])*(1./(n[i]*(n[i]+5)));
  // Boundary limit rules for the refined boundary edges.
  std::vector<V> bs(r->co.size());std::vector<int> bc(r->co.size());
  for(int e=0;e<(int)edges.size();++e)if(edges[e].f1<0){int m=nv+e;for(int a:{edges[e].a,edges[e].b}){bs[a]+=r->co[m];bc[a]++;bs[m]+=r->co[a];bc[m]++;}}
  for(int i=0;i<(int)out.size();++i)if(bc[i])out[i]=r->co[i]*(2./3)+bs[i]*(1./6);
  r->co.swap(out);
 }
 sizes[0]=r->co.size();sizes[1]=nl;return r;
}
extern "C" void subdivide_copy(void* ptr,float* co,int* quads){auto*r=static_cast<R*>(ptr);for(int i=0;i<(int)r->co.size();++i){co[3*i]=r->co[i].x;co[3*i+1]=r->co[i].y;co[3*i+2]=r->co[i].z;}std::memcpy(quads,r->faces.data(),r->faces.size()*4);delete r;}
