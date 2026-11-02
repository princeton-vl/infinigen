/*
COPYRIGHT


Original files authored by Nick McDonald: https://github.com/weigert/SoilMachine/tree/master


All other contributions:
Copyright (C) 2023, Princeton University.
Licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.
Authors: Zeyu Ma

*/

/*
================================================================================
                          Layermap Data Structure
================================================================================

Concept: Heightmap representet as a grid. Each grid element points to the "top"
element of a double-linked-list of RLE components which contain the layer meta data.
These components are additionally memory pooled for efficiency.

Utilizes a vertexpool to avoid remeshing when updating heightmap information

A layermap section has a height, a porosity and a water fraction.
The idea is that if the water fraction is 1, then all pores are filled.
This tells me the overall saturation of water inside.
Layermap has a height, a porosity and a

Air e.g. has porosity 1

*/
// #define SCALE 80
#include "include/FastNoiseLite.h"

#include <glm/glm.hpp>
#include <iostream>

using namespace glm;
using namespace std;

/*
================================================================================
                  RLE Linked-List Element and Memory Pool
================================================================================
*/

#include "surface.h"

struct sec {

sec* next = NULL;     //Element Above
sec* prev = NULL;     //Element Below

SurfType type = AIR;   //Type of Surface Element
double size = 0.0f;               //Run-Length of Element
double floor = 0.0f;              //Cumulative Height at Bottom
double saturation = 0.0f;         //Saturation with Water

sec(){}
sec(double s, SurfType t){
  size = s;
  type = t;
}

void reset(){
  next = NULL;
  prev = NULL;
  type = AIR;
  size = 0.0f;
  floor = 0.0f;
  saturation = 0.0f;
}

};

class secpool {
public:

int size;               //Number of Total Elements
sec* start = NULL;      //Point to Start of Pool
vector<sec*> free;      //Stack of Free Elements

secpool(){}             //Construct
secpool(const int N){   //Construct with Size
  reserve(N);
}
~secpool(){
  free.clear();
  delete[] start;
}

//Create the Memory Pool
void reserve(const int N){
  start = new sec[N];
  free.reserve(N);
  for(int i = 0; i < N; i++)
    free.push_back(start+i);
  size = N;
}

//Retrieve Element, Construct in Place
template<typename... Args>
sec* get(Args && ...args){

  if(free.empty()){
    cout<<"Memory Pool Out-Of-Elements"<<endl;
    return NULL;
  }

  sec* E = free.back();
  try{ new (E)sec(forward<Args>(args)...); }
  catch(...) { throw; }
  free.pop_back();
  return E;

}

//Return Element
void unget(sec* E){
  if(E == NULL)
    return;
  E->reset();
  free.push_back(E);
}

void reset(){
  free.clear();
  for(int i = 0; i < size; i++)
    free.push_back(start+i);
}

};

/*
================================================================================
                      Queriable Layermap Datastructure
================================================================================
*/

class Layermap {

private:

sec** dat = NULL;                         //Raw Data Grid
double* hcache = NULL;                    //Cached top height per cell

public:

int SCALE;
ivec2 dim;                                //Size
secpool pool;                             //Data Pool

//Queries
double height(ivec2);                     //Query Height at Discrete Position
void refresh_height(ivec2);               //Recompute cached height after add/remove
double saturation(ivec2);                     //Query saturation at Discrete Position
double height(vec2);                      //Query Height at Position (Bilinear Interpolation)
vec3 normal(ivec2);                       //Normal Vector at Position
vec3 normal(vec2);                        //Normal Vector at Position (Bilinear Interpolation)
vec3 normal(ivec2, Vertexpool<Vertex>&);  //Normal Vector at Position (Read from Vertexpool)
vec3 normal(vec2, Vertexpool<Vertex>&);   //Normal Vector at Position (Read from Vertexpool)
SurfType surface(ivec2);                  //Surface Type at Position

//Modifiers
void add(ivec2, sec*);                    //Add Layer at Position
double remove(ivec2, double);             //Remove Layer at Position
sec* top(ivec2 pos){                      //Top Element at Position
  return dat[pos.x*dim.y+pos.y];
}

//Meshing / Visualization
uint* section = NULL;                     //Vertexpool Section Pointer
// void meshpool(Vertexpool<Vertex>&);       //Mesh based on Vertexpool
// void update(ivec2, Vertexpool<Vertex>&);  //Update Vertexpool at Position (No Remesh)
// void update(Vertexpool<Vertex>&);         //Update Vertexpool at Position (No Remesh)
// void slice(Vertexpool<Vertex>&, double);         //Update Vertexpool at Position (No Remesh)

public:

void initialize(int SEED, ivec2 _dim, float *heightmap){

  dim = _dim;

  //Important so Re-Callable

  pool.reset();

  if(dat != NULL) delete[] dat;
  dat = new sec*[dim.x*dim.y];      //Array of Section Pointers
  if(hcache != NULL) delete[] hcache;
  hcache = new double[dim.x*dim.y];
  for(int i = 0; i < dim.x*dim.y; i++) hcache[i] = 0.0;

  for(int i = 0; i < dim.x; i++)
  for(int j = 0; j < dim.y; j++)
    dat[i*dim.y+j] = NULL;

  //Fill 'er up
  // cout << "layers.size()" << layers.size() << endl;
  
  const int MAXSEED = 10000;
  for(size_t l = 0; l < layers.size(); l++){

    const float f = (float)l/(float)layers.size();
    const int Z = SEED + f*MAXSEED;
    layers[l].init();

    //Add a first layer!
    for(int i = 0; i < dim.x; i++){
    for(int j = 0; j < dim.y; j++){

      // double h = layers[l].get(vec3(i, j, Z%MAXSEED)/vec3(dim.x, dim.y, 1));
      // modify it to be from heightmap
      double h = heightmap[i * dim.y + j] / layers.size();
      add(ivec2(i, j), pool.get(h, layers[l].type));

    }}

  }

}

~Layermap(){
  delete[] dat;
  delete[] hcache;
}

//Constructors
Layermap(int SEED, ivec2 _dim, int SCALE_, float* heightmap){
  pool.reserve(POOLSIZE);                //Some permissible amount of RAM later...
  initialize(SEED, _dim, heightmap);
  SCALE = SCALE_;
}

// Layermap(int SEED, ivec2 _dim, Vertexpool<Vertex>& vertexpool):Layermap(SEED, _dim){
//   meshpool(vertexpool);
// }

};

inline void Layermap::add(ivec2 pos, sec* E){

  //Non-Element: Don't Add
  if(E == NULL)
    return;

  //Negative Size Element: Don't Add
  if(E->size <= 0){
    pool.unget(E);
    return;
  }

  //Valid Element, Empty Spot: Set Top Directly
  if(dat[pos.x*dim.y+pos.y] == NULL){
    dat[pos.x*dim.y+pos.y] = E;
    refresh_height(pos);
    return;
  }

  //Valid Element, Previous Type Identical: Elongate
  if(dat[pos.x*dim.y+pos.y]->type == E->type){
    dat[pos.x*dim.y+pos.y]->size += E->size;
    pool.unget(E);
    refresh_height(pos);
    return;
  }

  //Basically: A position Swap

  //Add to Water, but not equal to water
  if(dat[pos.x*dim.y+pos.y]->type == AIR){ //Switch with Water

  //  pool.unget(E);

    //Remove Top Element (Water)
    sec* top = dat[pos.x*dim.y+pos.y];
    dat[pos.x*dim.y+pos.y] = top->prev;
    refresh_height(pos);

    //Add this Element
    add(pos, E);

    //Add Water Back In
    add(pos, top);


    return;

  }

  /*
  if(dat[pos.x*dim.y+pos.y]->prev != NULL)
  if(dat[pos.x*dim.y+pos.y]->prev->size < 0.01)
  if(dat[pos.x*dim.y+pos.y]->prev->type == E->type){    //Same Type: Make Taller, Remove E
    dat[pos.x*dim.y+pos.y]->prev->size += E->size;
    dat[pos.x*dim.y+pos.y]->floor += E->size;
    pool.unget(E);
    return;
  }
  */

  //Try a sorting move???

  /*
  if(dat[pos.x*dim.y+pos.y]->prev != NULL)
  if(soils[dat[pos.x*dim.y+pos.y]->type].density < soils[E->type].density)
  if(dat[pos.x*dim.y+pos.y]->prev->type == E->type){    //Same Type: Make Taller, Remove E
    dat[pos.x*dim.y+pos.y]->prev->size += E->size;
    dat[pos.x*dim.y+pos.y]->floor += E->size;
    pool.unget(E);
    return;
  }
  */

  //Add Element
  dat[pos.x*dim.y+pos.y]->next = E;
  E->prev = dat[pos.x*dim.y+pos.y];
  E->floor = height(pos);
  dat[pos.x*dim.y+pos.y] = E;
  refresh_height(pos);

}

//Returns Amount Removed
inline double Layermap::remove(ivec2 pos, double h){

  //No Element to Remove
  if(dat[pos.x*dim.y+pos.y] == NULL)
    return 0.0;

  //Element Needs Removal
  if(dat[pos.x*dim.y+pos.y]->size <= 0.0){
    sec* E = dat[pos.x*dim.y+pos.y];
    dat[pos.x*dim.y+pos.y] = E->prev; //May be NULL
    pool.unget(E);
    refresh_height(pos);
    return 0.0;
  }

  //No Removal Necessary (Note: Zero Height Elements Removed)
  if(h <= 0.0)
    return 0.0;

  double diff = h - dat[pos.x*dim.y+pos.y]->size;
  dat[pos.x*dim.y+pos.y]->size -= h;

  if(diff >= 0.0){
    sec* E = dat[pos.x*dim.y+pos.y];
    dat[pos.x*dim.y+pos.y] = E->prev; //May be NULL
    pool.unget(E);
    refresh_height(pos);
    return diff;
  }
  refresh_height(pos);
  return 0.0;

}

inline vec3 Layermap::normal(ivec2 pos){

  vec3 n = vec3(0);
  vec3 p = vec3(pos.x, SCALE*height(pos), pos.y);
  int k = 0;

  if(pos.x > 0 && pos.y > 0){
    vec3 b = vec3(pos.x-1, SCALE*height(pos-ivec2(1,0)), pos.y);
    vec3 c = vec3(pos.x, SCALE*height(pos-ivec2(0,1)), pos.y-1);
    n += cross(c-p, b-p);
    k++;
  }

  if(pos.x > 0 && pos.y < dim.y - 1){
    vec3 b = vec3(pos.x-1, SCALE*height(pos-ivec2(1,0)), pos.y);
    vec3 c = vec3(pos.x, SCALE*height(pos+ivec2(0,1)), pos.y+1);
    n -= cross(c-p, b-p);
    k++;
  }

  if(pos.x < dim.x-1 && pos.y > 0){
    vec3 b = vec3(pos.x+1, SCALE*height(pos+ivec2(1,0)), pos.y);
    vec3 c = vec3(pos.x, SCALE*height(pos-ivec2(0,1)), pos.y-1);
    n -= cross(c-p, b-p);
    k++;
  }

  if(pos.x < dim.x-1 && pos.y < dim.y-1){
    vec3 b = vec3(pos.x+1, SCALE*height(pos+ivec2(1,0)), pos.y);
    vec3 c = vec3(pos.x, SCALE*height(pos+ivec2(0,1)), pos.y+1);
    n += cross(c-p, b-p);
    k++;
  }

  return normalize(n/(float)k);

}

inline vec3 Layermap::normal(vec2 pos){

  vec3 n = vec3(0);
  ivec2 p = floor(pos);
  vec2 w = fract(pos);

  n += (1.0f-w.x)*(1.0f-w.y)*normal(p);
  n += (1.0f-w.x)*w.y*normal(p+ivec2(1,0));
  n += w.x*(1.0f-w.y)*normal(p+ivec2(0,1));
  n += w.x*w.y*normal(p+ivec2(1,1));

  return n;

}

vec3 Layermap::normal(ivec2 pos, Vertexpool<Vertex>& vertexpool){
  return normal(pos);
  //Vertex* v = vertexpool.get(section, pos.x*dim.y+pos.y);
  //return vec3(v->normal[0], v->normal[1], v->normal[2]);
}

vec3 Layermap::normal(vec2 pos, Vertexpool<Vertex>& vertexpool){

  vec3 n = vec3(0);
  ivec2 p = floor(pos);
  vec2 w = fract(pos);

  n += (1.0f-w.x)*(1.0f-w.y)*normal(p, vertexpool);
  n += (1.0f-w.x)*w.y*normal(p+ivec2(1,0), vertexpool);
  n += w.x*(1.0f-w.y)*normal(p+ivec2(0,1), vertexpool);
  n += w.x*w.y*normal(p+ivec2(1,1), vertexpool);

  return n;

}

//Queries

inline SurfType Layermap::surface(ivec2 pos){
  if(dat[pos.x*dim.y+pos.y] == NULL) return 0;
  return dat[pos.x*dim.y+pos.y]->type;
}

double Layermap::saturation(ivec2 pos){
  double ans = 0;
  auto data = dat[pos.x*dim.y+pos.y];
  while (data != NULL) {
    SurfParam param = soils[data->type];
    // if (data->type == AIR)
    ans += data->saturation * param.porosity * data->size;
    data = data->prev;
  }
  return ans;
}

inline double Layermap::height(ivec2 pos){
  return hcache[pos.x*dim.y+pos.y];
}

inline void Layermap::refresh_height(ivec2 pos){
  sec* t = dat[pos.x*dim.y+pos.y];
  hcache[pos.x*dim.y+pos.y] = (t == NULL) ? 0.0 : (t->floor + t->size);
}

inline double Layermap::height(vec2 pos){

  double h = 0.0f;
  ivec2 p = floor(pos);
  vec2 w = fract(pos);

  h += (1.0-w.x)*(1.0-w.y)*height(p);
  h += (1.0-w.x)*w.y*height(p+ivec2(1,0));
  h += w.x*(1.0-w.y)*height(p+ivec2(0,1));
  h += w.x*w.y*height(p+ivec2(1,1));
  return h;

}

// Meshing, Updating

// void Layermap::meshpool(Vertexpool<Vertex>& vertexpool){

//   if(section != NULL){
//     vertexpool.unsection(section);
//     vertexpool.indices.clear();
//   }

//   section = vertexpool.section(dim.x*dim.y, 0, glm::vec3(0));

//   for(int i = 0; i < dim.x; i++)
//   for(int j = 0; j < dim.y; j++)
//     update(ivec2(i,j), vertexpool);

//   for(int i = 0; i < dim.x-1; i++){
//   for(int j = 0; j < dim.y-1; j++){

//     vertexpool.indices.push_back(i*dim.y+j);
//     vertexpool.indices.push_back(i*dim.y+(j+1));
//     vertexpool.indices.push_back((i+1)*dim.y+j);

//     vertexpool.indices.push_back((i+1)*dim.y+j);
//     vertexpool.indices.push_back(i*dim.y+(j+1));
//     vertexpool.indices.push_back((i+1)*dim.y+(j+1));

//   }}

//   vertexpool.resize(section, vertexpool.indices.size());
//   vertexpool.index();
//   vertexpool.update();

// }

// void Layermap::update(ivec2 p, Vertexpool<Vertex>& vertexpool){

//   sec* top = dat[p.x*dim.y+p.y];
//   while(top != NULL && top->floor > (float)SLICE/(float)SCALE)
//     top = top->prev;

//   if(top == NULL){
//     vertexpool.fill(section, p.x*dim.y+p.y,
//       vec3(p.x, 0, p.y),
//       vec3(0,1,0),
//       soils[AIR].color,
//       AIR
//     );
//   }

//   else if(top->floor + top->size > (float)SLICE/(float)SCALE){

//     if(top->floor + top->size*top->saturation > (float)SLICE/(float)SCALE)
//     vertexpool.fill(section, p.x*dim.y+p.y,
//       vec3(p.x, SLICE, p.y),
//       vec3(0,1,0),
//   //    normal(p),
//       mix(soils[AIR].color, soils[top->type].color, 0.6),
//       AIR
//     );

//     else
//     vertexpool.fill(section, p.x*dim.y+p.y,
//       vec3(p.x, SLICE, p.y),
//       vec3(0,1,0),
// //    normal(p),
//       soils[top->type].color,
//       top->type
//     );

//   }

//   else{

// /*
//     if(top->saturation == 1)  //Fill Watertable!
//     vertexpool.fill(section, p.x*dim.y+p.y,
//       vec3(p.x, SCALE*(top->floor + top->size), p.y),
//       normal(p),
//       soils[AIR].color
//     );
// */
// //    else
//     vertexpool.fill(section, p.x*dim.y+p.y,
//       vec3(p.x, SCALE*(top->floor + top->size), p.y),
//       normal(p),
//       soils[top->type].color,
//       top->type
//     );

//   }

//   /*

//   if(surface(p) == AIR)
//   vertexpool.fill(section, p.x*dim.y+p.y,
//     vec3(p.x, SCALE*height(p), p.y),
//     normal(p),
//     soils[surface(p)].color
//   );
//   else
//   vertexpool.fill(section, p.x*dim.y+p.y,
//     vec3(p.x, SCALE*height(p), p.y),
//     normal(p),
//     soils[surface(p)].color
//   );

//   */

// }

// void Layermap::update(Vertexpool<Vertex>& vertexpool){
//   for(int i = 0; i < dim.x; i++)
//   for(int j = 0; j < dim.y; j++)
//     update(ivec2(i,j), vertexpool);
// }

// void Layermap::slice(Vertexpool<Vertex>& vertexpool, double s = SCALE){

//   for(int i = 0; i < dim.x; i++)
//   for(int j = 0; j < dim.y; j++){

//     ivec2 p = ivec2(i,j);

//     //Find the first element which starts below the scale!
//     sec* top = dat[p.x*dim.y+p.y];
//     while(top != NULL && top->floor > s/SCALE)
//       top = top->prev;

//     if(top == NULL){
//       vertexpool.fill(section, p.x*dim.y+p.y,
//         vec3(p.x, 0, p.y),
//         vec3(0,1,0),
//         soils[AIR].color,
//         AIR
//       );
//     }

//     else if(top->floor + top->size > s/SCALE){
//       if(top->floor + top->size*top->saturation > s/SCALE)
//       vertexpool.fill(section, p.x*dim.y+p.y,
//         vec3(p.x, s, p.y),
//         vec3(0,1,0),
//         mix(vec4(1,0,0,1), soils[top->type].color, 0.6),
//         top->type
//       );
//       else
//       vertexpool.fill(section, p.x*dim.y+p.y,
//         vec3(p.x, s, p.y),
//         vec3(0,1,0),
//         soils[top->type].color,
//         top->type
//       );
//     }

//     else{
//       if(top->saturation == 0)  //Fill Watertable!
//       vertexpool.fill(section, p.x*dim.y+p.y,
//         vec3(p.x, SCALE*(top->floor + top->size), p.y),
//         normal(p),
//         vec4(1,0,0,1),
//         top->type
//       );
//       else
//       vertexpool.fill(section, p.x*dim.y+p.y,
//         vec3(p.x, SCALE*(top->floor + top->size), p.y),
//         normal(p),
//         soils[top->type].color,
//         top->type
//       );
//     }

//   }
// }
