/*
COPYRIGHT


Original files authored by Nick McDonald: https://github.com/weigert/SoilMachine/tree/master


All other contributions:
Copyright (c) Princeton University.
Licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.
Authors: Zeyu Ma

*/

/*
================================================================================
        Particle Base Class for Geomorphological Transport Simulation
================================================================================
*/

#ifndef LAYEREDEROSION_PARTICLE
#define LAYEREDEROSION_PARTICLE

#include "../include/distribution.h"

using namespace glm;

struct Particle {

  vec2 pos;
  vec2 speed = vec2(0);
  bool isalive = true;

  bool move(Layermap& map);
  bool interact(Layermap& map, Vertexpool<Vertex>& vertexpool);

  // because I can't solve the bug, I'm not familiar with static in c++, so I make SCALE an arg --- Zeyu
  //This is applied to multiple types of erosion, so I put it in here!
  static void cascade(float SCALE, vec2 pos, Layermap& map, Vertexpool<Vertex>& vertexpool, int transferloop = 0){

    ivec2 ipos = round(pos);

    // All Possible Neighbors

    const vector<ivec2> n = {
      ivec2(-1, -1),
      ivec2(-1,  0),
      ivec2(-1,  1),
      ivec2( 0, -1),
      ivec2( 0,  1),
      ivec2( 1, -1),
      ivec2( 1,  0),
      ivec2( 1,  1)
    };

    //No Out-Of-Bounds

    vector<ivec2> sn;
    for(auto& nn: n){
      ivec2 npos = ipos + nn;
      if(npos.x >= map.dim.x || npos.y >= map.dim.y
         || npos.x < 0 || npos.y < 0) continue;
      sn.push_back(npos);
    }

    // Sort by Highest First (Soil is Moved Down After All)

    sort(sn.begin(), sn.end(), [&](const ivec2& a, const ivec2& b){
      return map.height(a) > map.height(b);
    });

    for(auto& npos: sn){

      //Full Height-Different Between Positions!
      float diff = (map.height(ipos) - map.height(npos))*(float)SCALE/80.0f;

      if(diff == 0)   //No Height Difference
        continue;

      ivec2 tpos = (diff > 0) ? ipos : npos;
      ivec2 bpos = (diff > 0) ? npos : ipos;

      SurfType type = map.surface(tpos);
      SurfParam param = soils[type];

      //The Amount of Excess Difference!
      float excess = abs(diff) - param.maxdiff;
      if(excess <= 0)  //No Excess
        continue;

      //Actual Amount Transferred
      float transfer = param.settling * excess / 2.0f;

      bool recascade = false;

      if(transfer > map.top(tpos)->size)
        transfer = map.top(tpos)->size;

      if(map.remove(tpos, transfer) != 0)
        recascade = true;
      map.add(bpos, map.pool.get(transfer, param.cascades));
      // map.update(tpos, vertexpool);
      // map.update(bpos, vertexpool);

      if(recascade && transferloop > 0)
        cascade(SCALE, npos, map, vertexpool, --transferloop);

    }

  }

};

#endif
