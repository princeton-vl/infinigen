// SPDX-License-Identifier: BSD-3-Clause
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <cstring>
extern "C" int pinecone_edges(const int* loops,const int* counts,int nf,int* edges,int* loop_edges){
 std::unordered_map<uint64_t,int> map;int offset=0;for(int f=0;f<nf;++f)offset+=counts[f];map.reserve(offset/2);offset=0;int ne=0;
 for(int f=0;f<nf;++f){int size=counts[f];for(int j=0;j<size;++j){int a=loops[offset+j],b=loops[offset+(j+1)%size];if(a>b)std::swap(a,b);uint64_t key=(uint64_t(a)<<32)|uint32_t(b);auto [it,added]=map.emplace(key,ne);if(added){edges[2*ne]=a;edges[2*ne+1]=b;++ne;}loop_edges[offset+j]=it->second;}offset+=size;}return ne;
}
