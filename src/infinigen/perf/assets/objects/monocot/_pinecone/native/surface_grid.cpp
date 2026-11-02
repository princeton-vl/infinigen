// SPDX-License-Identifier: BSD-3-Clause
// Surface-only grid intersection meshing: no signed distance field or volume fill.
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <cstdint>
#include <set>
#include <chrono>
using P=std::array<float,3>;using F=std::array<int,3>;
struct Mesh{double seconds[3];std::vector<P>co;std::vector<F>tri;std::vector<std::array<int,2>>edges;std::vector<F>face_edges;};
struct Hit{double t;int tri;bool positive;};
struct Cross{std::array<double,3>point;int triangle;bool inside;};
struct Cell{std::array<int,12>edge;Cell(){edge.fill(-1);}};
static double dot(const std::array<double,3>&a,const std::array<double,3>&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
using D=std::array<double,3>;
static inline D sub(D a,D b){for(int i=0;i<3;++i)a[i]-=b[i];return a;}
static inline D add(D a,D b,double s){for(int i=0;i<3;++i)a[i]+=s*b[i];return a;}
static inline D closest(D p,D a,D b,D c){D ab=sub(b,a),ac=sub(c,a),ap=sub(p,a);double d1=dot(ab,ap),d2=dot(ac,ap);D bp=sub(p,b);double d3=dot(ab,bp),d4=dot(ac,bp);D cp=sub(p,c);double d5=dot(ab,cp),d6=dot(ac,cp);double vc=d1*d4-d3*d2,vb=d5*d2-d1*d6,va=d3*d6-d5*d4;
 if(d1<=0&&d2<=0)return a;if(d3>=0&&d4<=d3)return b;if(vc<=0&&d1>=0&&d3<=0)return add(a,ab,d1/(d1-d3));if(d6>=0&&d5<=d6)return c;if(vb<=0&&d2>=0&&d6<=0)return add(a,ac,d2/(d2-d6));if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0)return add(b,sub(c,b),(d4-d3)/((d4-d3)+(d5-d6)));if(std::abs(va+vb+vc)<1e-30)return a;return add(add(a,ab,vb/(va+vb+vc)),ac,vc/(va+vb+vc));}
static D cross(D a,D b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
static double triangle_quality(D a,D b,D c){auto x=sub(b,a),y=sub(c,b),z=sub(a,c);double ss=dot(x,x)+dot(y,y)+dot(z,z);D n=cross(x,y);return dot(n,n)/(ss*ss+1e-100);}

// Per-case contour connectivity is independent of scale geometry.
struct Pattern{std::array<std::array<int,12>,4>loops{};std::array<int,4>length{};int count=0;};
struct Cases{
 int ends[12][2],pair[8][8],face_mask[12]={},sign_for_edges[4096];Pattern patterns[256];
 Cases(){
  std::fill(&pair[0][0],&pair[0][0]+64,-1);std::fill(sign_for_edges,sign_for_edges+4096,-1);
  for(int axis=0;axis<3;++axis)for(int u=0;u<2;++u)for(int v=0;v<2;++v){int e=axis*4+2*u+v,a=(u<<((axis+1)%3))+(v<<((axis+2)%3)),b=a+(1<<axis);ends[e][0]=a;ends[e][1]=b;pair[a][b]=pair[b][a]=e;}
  for(int e=0;e<12;++e)for(int axis=0;axis<3;++axis)if((ends[e][0]&(1<<axis))==(ends[e][1]&(1<<axis)))face_mask[e]|=1<<(axis*2+((ends[e][0]>>axis)&1));
  for(int mask=0;mask<256;++mask){int edge_mask=0;for(int e=0;e<12;++e)if(((mask>>ends[e][0])^(mask>>ends[e][1]))&1)edge_mask|=1<<e;if(!(mask&1))sign_for_edges[edge_mask]=mask;
   int adj[12][2]={},degree[12]={};auto connect=[&](int a,int b){adj[a][degree[a]++]=b;adj[b][degree[b]++]=a;};
   for(int axis=0;axis<3;++axis)for(int side=0;side<2;++side){int base=side<<axis,u=1<<((axis+1)%3),v=1<<((axis+2)%3);int corners[4]={base,base+u,base+u+v,base+v},es[4],active[4],n=0;
    for(int j=0;j<4;++j){es[j]=pair[corners[j]][corners[(j+1)%4]];if(edge_mask&(1<<es[j]))active[n++]=es[j];}
    if(n==2)connect(active[0],active[1]);else if(n==4)for(int j=0;j<4;++j)if(mask&(1<<corners[j]))connect(es[(j+3)%4],es[j]);
   }
   bool used[12]={};auto&pattern=patterns[mask];for(int start=0;start<12;++start)if((edge_mask&(1<<start))&&!used[start]){int prev=-1,cur=start,n=0;do{used[cur]=true;pattern.loops[pattern.count][n++]=cur;int next=adj[cur][0]==prev?adj[cur][1]:adj[cur][0];prev=cur;cur=next;}while(cur!=start);pattern.length[pattern.count++]=n;}
  }
 }
};
static const Cases cases;
struct FlatIndex{
 std::vector<uint64_t>keys;std::vector<int>values;size_t mask;
 explicit FlatIndex(size_t n){size_t size=16;while(size<n*2)size*=2;keys.assign(size,UINT64_MAX);values.resize(size);mask=size-1;}
 int&get(uint64_t key){uint64_t h=key;h^=h>>30;h*=0xbf58476d1ce4e5b9ULL;h^=h>>27;h*=0x94d049bb133111ebULL;h^=h>>31;size_t i=h&mask;while(keys[i]!=UINT64_MAX&&keys[i]!=key)i=(i+1)&mask;if(keys[i]==UINT64_MAX){keys[i]=key;values[i]=-1;}return values[i];}
};
extern "C" void* pinecone_surface(const float*xyz,int nv,const int*tri,int nt,float step,int*sizes){
 auto begin=std::chrono::steady_clock::now();
 const double shift[3]={.000031,.000053,.000079};auto*out=new Mesh;
 std::vector<Cross>crossings;std::vector<Cell>cells;std::unordered_map<uint64_t,int>index;
 std::vector<D>grid(nv);int lo[3],dims[3];size_t volume=1;
 for(int k=0;k<3;++k){double lower=1e30,upper=-1e30;for(int i=0;i<nv;++i){grid[i][k]=double(xyz[3*i+k])/step-shift[k];lower=std::min(lower,grid[i][k]);upper=std::max(upper,grid[i][k]);}lo[k]=int(std::floor(lower))-1;dims[k]=int(std::ceil(upper))-lo[k]+2;volume*=dims[k];}
 std::vector<int>dense_index;if(volume<=4000000)dense_index.assign(volume,-1);
 crossings.reserve(8192);cells.reserve(8192);out->co.reserve(8192);out->tri.reserve(16384);
 auto cell=[&](int*p){if(!dense_index.empty()){size_t key=(size_t(p[0]-lo[0])*dims[1]+p[1]-lo[1])*dims[2]+p[2]-lo[2];int&id=dense_index[key];if(id<0){id=cells.size();cells.emplace_back();}return id;}uint64_t k=(uint64_t(p[0]+524288)<<40)|(uint64_t(p[1]+524288)<<20)|uint64_t(p[2]+524288);auto it=index.find(k);if(it!=index.end())return it->second;int id=cells.size();cells.emplace_back();index[k]=id;return id;};
 for(int axis=0;axis<3;++axis){int u=(axis+1)%3,v=(axis+2)%3;double umin=1e30,umax=-1e30,vmin=1e30,vmax=-1e30;
  for(int i=0;i<nv;++i){umin=std::min(umin,double(xyz[i*3+u])/step);umax=std::max(umax,double(xyz[i*3+u])/step);vmin=std::min(vmin,double(xyz[i*3+v])/step);vmax=std::max(vmax,double(xyz[i*3+v])/step);}
  int lo_u=std::floor(umin)-1,lo_v=std::floor(vmin)-1,nu=int(std::ceil(umax))-lo_u+2,nv2=int(std::ceil(vmax))-lo_v+2;
  struct HitNode{Hit hit;int next=-1;};std::vector<HitNode>nodes;nodes.reserve(8192);std::vector<int>heads(nu*nv2,-1),tails(nu*nv2,-1);
  auto add_hit=[&](int ray,Hit hit){int id=nodes.size();nodes.push_back({hit,-1});if(tails[ray]>=0)nodes[tails[ray]].next=id;else heads[ray]=id;tails[ray]=id;};
  for(int i=0;i<nt;++i){double p[3][3];for(int j=0;j<3;++j)for(int c=0;c<3;++c)p[j][c]=grid[tri[i*3+j]][c];
   double ax=p[1][u]-p[0][u],ay=p[1][v]-p[0][v],bx=p[2][u]-p[0][u],by=p[2][v]-p[0][v],det=ax*by-ay*bx;if(std::abs(det)<1e-15)continue;
   int x0=std::ceil(std::min({p[0][u],p[1][u],p[2][u]})),x1=std::floor(std::max({p[0][u],p[1][u],p[2][u]}));int y0=std::ceil(std::min({p[0][v],p[1][v],p[2][v]})),y1=std::floor(std::max({p[0][v],p[1][v],p[2][v]}));
   for(int x=x0;x<=x1;++x)for(int y=y0;y<=y1;++y){double dx=x-p[0][u],dy=y-p[0][v],s=(dx*by-dy*bx)/det,t=(ax*dy-ay*dx)/det;if(s<0||t<0||s+t>=1)continue;double z=p[0][axis]+s*(p[1][axis]-p[0][axis])+t*(p[2][axis]-p[0][axis]);add_hit((x-lo_u)*nv2+y-lo_v,{z,i,det>0});}
  }
  std::vector<Hit>h;h.reserve(32);
  for(int ray=0;ray<int(heads.size());++ray){h.clear();for(int id=heads[ray];id>=0;id=nodes[id].next)h.push_back(nodes[id].hit);std::sort(h.begin(),h.end(),[](Hit a,Hit b){return a.t<b.t;});int x=ray/nv2+lo_u,y=ray%nv2+lo_v;
   for(int i=0;i<int(h.size());){int j=i+1,z=std::floor(h[i].t);while(j<int(h.size())&&std::floor(h[j].t)==z)++j;
    if((j-i)%2){Hit hit=h[i];D point;point[axis]=hit.t;point[u]=x;point[v]=y;for(int k=0;k<3;++k)point[k]=(point[k]+shift[k])*step;
     int id=crossings.size();crossings.push_back({point,hit.tri,bool(i%2)});out->co.push_back({float(point[0]),float(point[1]),float(point[2])});
     for(int du=0;du<2;++du)for(int dv=0;dv<2;++dv){int p[3];p[axis]=z;p[u]=x-du;p[v]=y-dv;cells[cell(p)].edge[axis*4+du*2+dv]=id;}
    }i=j;
   }
  }
 }
 auto raster_end=std::chrono::steady_clock::now();
 const auto&face_mask=cases.face_mask;
 int bad=0,fallback=0;
 // Normalize each source normal once.
 std::vector<D>normals(nt);for(int t=0;t<nt;++t){D p[3];for(int j=0;j<3;++j)for(int k=0;k<3;++k)p[j][k]=xyz[tri[t*3+j]*3+k];D n=cross(sub(p[1],p[0]),sub(p[2],p[0]));double length=std::sqrt(dot(n,n));normals[t]=length>0?add(D{0,0,0},n,1/length):D{0,0,0};}
 for(auto&cell:cells){int edge_mask=0,first=-1;for(int e=0;e<12;++e)if(cell.edge[e]>=0){edge_mask|=1<<e;if(first<0)first=e;}
  int mask=cases.sign_for_edges[edge_mask];if(mask<0||first<0){++bad;continue;}
  if(((mask>>cases.ends[first][0])&1)!=crossings[cell.edge[first]].inside)mask^=255;
  bool valid=true;for(int e=0;e<12;++e)if(cell.edge[e]>=0&&((mask>>cases.ends[e][0])&1)!=crossings[cell.edge[e]].inside)valid=false;
  if(!valid){++bad;continue;}
  const auto&pattern=cases.patterns[mask];
  for(int loop=0;loop<pattern.count;++loop){std::vector<int>poly;poly.reserve(12);D normal={0,0,0};
   for(int k=0;k<pattern.length[loop];++k){int id=cell.edge[pattern.loops[loop][k]];poly.push_back(id);normal=add(normal,normals[crossings[id].triangle],1);}
   D area={0,0,0};for(int j=0;j<int(poly.size());++j)area=add(area,cross(crossings[poly[j]].point,crossings[poly[(j+1)%poly.size()]].point),1);
   if(dot(area,normal)<0)std::reverse(poly.begin(),poly.end());
   auto same_face=[&](int a,int b){int ea=-1,eb=-1;for(int e=0;e<12;++e){if(cell.edge[e]==a)ea=e;if(cell.edge[e]==b)eb=e;}return ea>=0&&eb>=0&&(face_mask[ea]&face_mask[eb]);};
   while(poly.size()>3){int best=-1;double score=-1;int n=poly.size();
    for(int j=0;j<n;++j){if(same_face(poly[(j+n-1)%n],poly[(j+1)%n]))continue;D a=crossings[poly[(j+n-1)%n]].point,b=crossings[poly[j]].point,c=crossings[poly[(j+1)%n]].point;if(dot(cross(sub(b,a),sub(c,b)),normal)<=0)continue;bool contains=false;
     for(int k=0;k<n;++k)if(k!=j&&k!=(j+n-1)%n&&k!=(j+1)%n){D p=crossings[poly[k]].point;if(dot(cross(sub(b,a),sub(p,a)),normal)>=0&&dot(cross(sub(c,b),sub(p,b)),normal)>=0&&dot(cross(sub(a,c),sub(p,c)),normal)>=0){contains=true;break;}}
     double q=triangle_quality(a,b,c);if(!contains&&q>score){score=q;best=j;}
    }
    if(best<0){++fallback;D p={0,0,0};for(int id:poly)p=add(p,crossings[id].point,1./poly.size());int center=out->co.size();out->co.push_back({float(p[0]),float(p[1]),float(p[2])});for(int j=0;j<int(poly.size());++j)out->tri.push_back({poly[j],poly[(j+1)%poly.size()],center});poly.clear();break;}
    int n2=poly.size();out->tri.push_back({poly[(best+n2-1)%n2],poly[best],poly[(best+1)%n2]});poly.erase(poly.begin()+best);
   }
   if(poly.size()==3)out->tri.push_back({poly[0],poly[1],poly[2]});
  }
 }

 // Orient each connected surface consistently; triangle-loop geometry can be warped.
 auto contours_end=std::chrono::steady_clock::now();
 struct Adj{int f=-1,g=-1;bool direction=false,same=false;};
 std::vector<int>edge_head(out->co.size(),-1),edge_next;edge_next.reserve(out->tri.size()*2);std::vector<Adj>adj;adj.reserve(out->tri.size()*2);out->edges.reserve(out->tri.size()*2);out->face_edges.resize(out->tri.size());
 auto edgekey=[](int a,int b){if(a>b)std::swap(a,b);return (uint64_t(a)<<32)|uint32_t(b);};
 for(int f=0;f<int(out->tri.size());++f)for(int j=0;j<3;++j){int a=out->tri[f][j],b=out->tri[f][(j+1)%3],low=std::min(a,b),high=std::max(a,b),id=edge_head[low];while(id>=0&&out->edges[id][1]!=high)id=edge_next[id];if(id<0){id=adj.size();adj.push_back({f,-1,a<b,false});out->edges.push_back({low,high});edge_next.push_back(edge_head[low]);edge_head[low]=id;}else{adj[id].g=f;adj[id].same=adj[id].direction==(a<b);}out->face_edges[f][j]=id;}

 std::vector<int>flip(out->tri.size(),-1),component;component.reserve(out->tri.size());
 for(int seed=0;seed<int(out->tri.size());++seed)if(flip[seed]<0){component.clear();component.push_back(seed);flip[seed]=0;
  for(size_t i=0;i<component.size();++i){int f=component[i];for(int j=0;j<3;++j){auto ad=adj[out->face_edges[f][j]];int g=ad.f==f?ad.g:ad.f;if(g>=0&&flip[g]<0){flip[g]=flip[f]^ad.same;component.push_back(g);}}}
  double volume=0;for(int f:component){if(flip[f]){std::swap(out->tri[f][1],out->tri[f][2]);std::swap(out->face_edges[f][0],out->face_edges[f][2]);}D p[3];for(int j=0;j<3;++j)for(int k=0;k<3;++k)p[j][k]=out->co[out->tri[f][j]][k];volume+=dot(p[0],cross(p[1],p[2]));}
  if(volume<0)for(int f:component){std::swap(out->tri[f][1],out->tri[f][2]);std::swap(out->face_edges[f][0],out->face_edges[f][2]);}
 }
 auto finish=std::chrono::steady_clock::now();out->seconds[0]=std::chrono::duration<double>(raster_end-begin).count();out->seconds[1]=std::chrono::duration<double>(contours_end-raster_end).count();out->seconds[2]=std::chrono::duration<double>(finish-contours_end).count();
 sizes[0]=out->co.size();sizes[1]=out->tri.size();sizes[2]=bad;sizes[3]=fallback;return out;
}
extern "C" void pinecone_surface_copy(void*h,float*co,int*tri){auto*r=(Mesh*)h;std::copy((float*)r->co.data(),(float*)r->co.data()+r->co.size()*3,co);std::copy((int*)r->tri.data(),(int*)r->tri.data()+r->tri.size()*3,tri);delete r;}

extern "C" int pinecone_surface_edge_count(void*h){return ((Mesh*)h)->edges.size();}
extern "C" void pinecone_surface_connectivity(void*h,int*edges,int*le){auto*r=(Mesh*)h;std::copy((int*)r->edges.data(),(int*)r->edges.data()+r->edges.size()*2,edges);std::copy((int*)r->face_edges.data(),(int*)r->face_edges.data()+r->face_edges.size()*3,le);}

extern "C" void pinecone_surface_phases(void*h,double*out){std::copy(((Mesh*)h)->seconds,((Mesh*)h)->seconds+3,out);}
