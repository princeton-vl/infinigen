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
                  Water Particle for Hydraulic Erosion
================================================================================
*/
// #define SCALE 80
#include "particle.h"

struct WaterParticle : public Particle {

  WaterParticle(Layermap& map, float SCALE_){

    pos = vec2(rand()%map.dim.x, rand()%map.dim.y);
    ipos = round(pos);
    surface = map.surface(ipos);
    param = soils[surface];
    contains = param.transports;    //The Transporting Type
    SCALE = SCALE_;
  }

  static void init(int SIZEX, int SIZEY){
    frequency = new float[SIZEX*SIZEY]{0.0f};
    track = new float[SIZEX*SIZEY]{0.0f};
    // SCALE = SCALE_;
  }

  //Core Properties
  double volume = 1.0;   //This will vary in time
  double sediment = 0.0; //Fraction of Volume that is Sediment!

  const double minvol = 0.01;
  double evaprate = 0.001;

  static double volumeFactor;
  int spill = 3;

  //Helper Properties
  ivec2 ipos;
  vec3 n;
  SurfParam param;
  SurfType surface;
  SurfType contains;

  static float* frequency;
  static float* track;
  float SCALE;


  void updatefrequency(Layermap& map, ivec2 ipos){
    int ind = ipos.y*map.dim.x+ipos.x;
    track[ind] += volume;
  }

  static void resetfrequency(Layermap& map){
    for(int i = 0; i < map.dim.x*map.dim.y; i++)
      track[i] = 0.0f;
  }

  static void mapfrequency(Layermap& map){
    const float lrate = 0.01f;
    const float K = 50.0f;
//    const float lrate = 0.05f;
//    const float K = 15.0f;
    for(int i = 0; i < map.dim.x*map.dim.y; i++)
      frequency[i] = (1.0f-lrate)*frequency[i] + lrate*K*track[i]/(1.0f + K*track[i]);;
  }

  bool move(Layermap& map, Vertexpool<Vertex>& vertexpool){

    ipos = round(pos);                //Position
    n = map.normal(ipos);             //Surface Normal Vector
    surface = map.surface(ipos);      //Surface Composition
    param = soils[surface];           //Surface Composition
    evaprate = 0.01;                 //Reset Evaprate
    updatefrequency(map, ipos);

    //Modify Parameters Based on Frequency
    param.friction = param.friction*(1.0f-frequency[ipos.y*map.dim.x+ipos.x]);
    evaprate = evaprate*(1.0f-0.2f*frequency[ipos.y*map.dim.x+ipos.x]);

    if(length(vec2(n.x, n.z)*param.friction) < 1E-5)   //No Motion
      return false;

    //Motion Low
    // speed = mix(vec2(n.x, n.z), speed, param.friction);
    speed = vec2(n.x, n.z) * float(1 - param.friction) + speed * float(param.friction);
    speed = sqrt(2.0f)*normalize(speed);
    pos   += speed;

    //Out-of-Bounds
    if(!glm::all(glm::greaterThanEqual(pos, vec2(0))) ||
       !glm::all(glm::lessThan(pos, (vec2)map.dim-1.0f))){
         volume = 0.0;
         return false;
       }

    return true;

  }

  bool interact(Layermap& map, Vertexpool<Vertex>& vertexpool){

    //Equilibrium Sediment Transport Amount
    double c_eq = param.solubility*(map.height(ipos)-map.height(pos))*(double)SCALE/80.0;
    if(c_eq < 0.0) c_eq = 0.0;
    if(c_eq > 1.0) c_eq = 1.0;

    //Erode Sediment IN Particle
    if((double)(soils[contains].erosionrate) < frequency[ipos.y*map.dim.x+ipos.x])
      contains = soils[contains].erodes;

    //Execute Transport to Particle
    double cdiff = c_eq - sediment;

    //Remove Sediment from Map

    if(cdiff > 0) {

      sediment += param.equrate*cdiff;
      contains = soils[map.surface(ipos)].transports;
  //    if(volume > 1) volume = 1;
      double diff = map.remove(ipos, param.equrate*cdiff*volume);
      while(abs(diff) > 1E-8){
        diff = map.remove(ipos, diff);
      }

    }

    //Add Sediment to Map

    else if(cdiff < 0) {

      sediment += soils[contains].equrate*cdiff;
      map.add(ipos, map.pool.get(-soils[contains].equrate*cdiff*volume, contains));

    }

    //Particle Cascade: Thermal Erosion!
    Particle::cascade(SCALE, pos, map, vertexpool, 0);

    //Update Map, Particle
    sediment /= (1.0-evaprate);
    if(sediment > 1.0) sediment = 1.0;
    volume *= (1.0-evaprate);
    return (volume > minvol);

  }

  bool flood(Layermap& map, Vertexpool<Vertex>& vertexpool){

    if(volume < minvol || spill-- <= 0)
      return false;

    ipos = pos;

    // Add Water

    // Add Remaining Soil

    map.add(ipos, map.pool.get(sediment*soils[contains].equrate, contains));
    Particle::cascade(SCALE, pos, map, vertexpool, 0);

    map.add(ipos, map.pool.get(volume*volumeFactor, soilmap["Air"]));
    seep(ipos, map, vertexpool);
    WaterParticle::cascade(SCALE, ipos, map, vertexpool, spill);

    // map.update(ipos, vertexpool);
    return false;

  }





  static void cascade(float SCALE, vec2 pos, Layermap& map, Vertexpool<Vertex>& vertexpool, int spill = 0){

    ivec2 ipos = pos;

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

    vector<ivec2> sn;
    for(auto& nn: n){
      ivec2 npos = ipos + nn;
      if(npos.x >= map.dim.x || npos.y >= map.dim.y
         || npos.x < 0 || npos.y < 0) continue;
      sn.push_back(npos);
    }

    // Sort by Steepness
    sort(sn.begin(), sn.end(), [&](const ivec2& a, const ivec2& b){
      return map.height(a) > map.height(b);
    });

    for(auto& npos: sn){

      sec* secA = map.top(ipos);
      sec* secB = map.top(npos);

      // Water Table Heights
      double whA = 0, whB = 0;
      if(secA != NULL){
        if(secA->type == soilmap["Air"])
        whA = secA->size;//*soils[secA->type].porosity*secA->saturation;
        else whA = secA->size;
      }
      if(secB != NULL){
        if(secB->type == soilmap["Air"])
        whB = secB->size;//*soils[secB->type].porosity*secB->saturation;
        else whB = secB->size;
      }

      // Floor Values
      double fA = 0.0;
      double fB = 0.0;

      if(secA != NULL)
        fA = secA->floor;

      if(secB != NULL)
        fB = secB->floor;

      // Actual Height Difference Between Watertables
      double diff = (fA + whA - fB - whB)*(double)SCALE/80.0;
      if(diff == 0)   //No Height Difference
        continue;

      // Higher and Lower Section
      sec* top = (diff > 0)?secA:secB;
      sec* bot = (diff > 0)?secB:secA;

      ivec2 tpos = (diff > 0) ? ipos : npos;
      ivec2 bpos = (diff > 0) ? npos : ipos;

      // We are currently only cascading air
      if(top->type != soilmap["Air"])
        continue;

      //Maximum Transferrable Amount of Water (Height Difference)
      double transfer = abs(diff) / 2.0;

      //Actual Amount of Water Available
      double wh = top->size;
      transfer = (wh < transfer) ? wh : transfer;

      if(transfer <= 0)
        continue;

      //we are cascading an amount which is equal to our volume!

      bool recascade = false;

      if(transfer == wh){

        double diff = map.remove(tpos, transfer);
        // map.update(tpos, vertexpool);

        WaterParticle particle(map, SCALE);
        particle.speed = sqrt(2.0f)*normalize(glm::vec2(bpos)-glm::vec2(tpos));

        particle.pos = tpos;
        particle.spill = spill;
        particle.volume = transfer / WaterParticle::volumeFactor;

        while(true){
          while(particle.move(map, vertexpool) && particle.interact(map, vertexpool));
          if(!particle.flood(map, vertexpool))
            break;
        }

      }

      else {

        if(map.remove(tpos, transfer) != 0)
          recascade = true;
        if(transfer > 0) recascade = true;
        map.add(bpos, map.pool.get(transfer, soilmap["Air"]));
        map.top(bpos)->saturation = 1.0f;
        // map.update(tpos, vertexpool);
        // map.update(bpos, vertexpool);

      }

      if(recascade && spill > 0)
        WaterParticle::cascade(SCALE, npos, map, vertexpool, --spill);

    }

  }

  static void seep(vec2 pos, Layermap& map, Vertexpool<Vertex>& vertexpool){

    ivec2 ipos = pos;

    sec* top = map.top(ipos);
    double pressure = 0.0f;            //Pressure Increases Moving Down
    if(top == NULL) return;

    while(top != NULL && top->prev != NULL){

      sec* prev = top->prev;

      SurfParam param = soils[top->type];
      SurfParam nparam = soils[prev->type];

      // Volume Top Layer
      double vol = top->size*top->saturation*param.porosity;

      //Volume Bottom Layer
      double nvol = prev->size*prev->saturation*nparam.porosity;

      //Empty Volume Bottom Layer
      double nevol = prev->size*(1.0 - prev->saturation)*nparam.porosity;

      double seepage = 1.0;//0.1f;

      // Compute Pressure
      //pressure *= (1.0f - param.porosity);  //Pressure Drop
      //pressure += vol;                      //Increase
      //Seepage Rate (Top-To-Bottom)
    //	double seepage = 1.0f / (1.0f + pressure * );

      //Transferred Volume is the Smaller Amount!
      double transfer = (vol < nevol) ? vol : nevol;
      if(transfer < 1E-6) seepage = 1.0; //Just Remove the Rest

      if(transfer > 0){

        // Remove from Top Layer
        if(top->type == soilmap["Air"])
          double diff = map.remove(ipos, seepage*transfer);
        else
          top->saturation -= (seepage*transfer) / (top->size*param.porosity);

        prev->saturation += (seepage*transfer) / (prev->size*nparam.porosity);

      }

      top = prev;

    }

    // map.update(ipos, vertexpool);

  }

  static void seep(float SCALE, Layermap& map, Vertexpool<Vertex>& vertexpool){

    for(size_t x = 0; x < map.dim.x; x++)
    for(size_t y = 0; y < map.dim.y; y++){
      seep(ivec2(x,y), map, vertexpool);
      WaterParticle::cascade(SCALE, ivec2(x,y), map, vertexpool, 3);
    }

  }

};

double WaterParticle::volumeFactor = 0.005;

float* WaterParticle::frequency = NULL;//new float[SIZEX*SIZEY]{0.0f};
float* WaterParticle::track = NULL;//new float[SIZEX*SIZEY]{0.0f};
