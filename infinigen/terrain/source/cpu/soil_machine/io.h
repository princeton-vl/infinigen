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
                Loading and Saving of SoilMachine Data
================================================================================
*/

#include <fstream>
#include <iostream>

void loadsoil(char* file_char){
  string file = file_char;
  ifstream in(file, ios::in);
  if(!in.is_open()){
    cout<<"Error: Failed to open soil profile "<<file<<endl;
    exit(0);
  }

  string line;
  int linenr = 0;

  const function<void()> syntaxerr = [&](){
    cout<<"Error: Incorrect Syntax in Line "<<linenr<<endl;
    exit(0);
  };

  const function<vec4(string)> hexcol = [&](string h){
    if(h.size() < 6) syntaxerr(); //Insufficient Characters
    const string allowed = "0123456789ABCDEF";
    for(auto& c: h) //Invalid Character Check
      if(allowed.find(c) == string::npos)
        syntaxerr();
    float R = 16*allowed.find(h[0]) + allowed.find(h[1]);
    float G = 16*allowed.find(h[2]) + allowed.find(h[3]);
    float B = 16*allowed.find(h[4]) + allowed.find(h[5]);
    return vec4(R, G, B, 255.0)/255.0f;
  };

  SurfParam param;                  //New Soil Class

  bool open = false;
  string soillayer;

  while (getline(in, line)){
    linenr++;

    size_t found = line.find('#');
    if(found != string::npos)       //Comment in line
      line = line.substr(0, found); //Comment removed

    if(line == "") continue;        //Empty Lines, Full Comments

    //We finish a guy
    if(line == "}"){
      if(!open) syntaxerr();
      if(soillayer == "SOIL"){
        // cout<<"Adding Soil Type "<<param.name<<endl;
        soils[soilmap[param.name]] = param;
      }
      open = false;
      continue;
    }

    //Handle Lines
    found = line.find(' ');         //Find Split
    if(found == string::npos)       //Comment in line
      syntaxerr();

    string tag = line.substr(0, found);
    string val = line.substr(found+1);

    if(tag == "SOIL"){

      found = val.find('{');
      if(found == string::npos)    //Missing Opening Bracket
        syntaxerr();

      param.name = val.substr(0, found-1);

      if(!soilmap.count(param.name)){
        soilmap[param.name] = soils.size(); //New Soil Number
        soils.push_back(param);             //Add Soil to Back
      }

      soillayer = tag;
      open = true;
      continue;

    }


    if(tag == "LAYER"){

      found = val.find('{');
      if(found == string::npos)    //Missing Opening Bracket
        syntaxerr();

      param.name = val.substr(0, found-1);

      if(!soilmap.count(param.name)){
        cout<<"Can't find SOIL "<<param.name<<endl;
        syntaxerr();
      }

      //Add the Layer
      // cout<<"Adding Layer Type "<<param.name<<endl;
      layers.emplace_back(soilmap[param.name]);

      soillayer = tag;
      open = true;
      continue;

    }

    if(tag == "WORLD"){

      found = val.find('{');
      if(found == string::npos)    //Missing Opening Bracket
        syntaxerr();

      soillayer = tag;
      open = true;
      continue;

    }

    if(soillayer == "SOIL"){

      if(tag == "TRANSPORTS"){
        if(!soilmap.count(val)){
          soilmap[val] = soils.size(); //New Soil Number
          soils.push_back(param);             //Add Soil to Back
        }
        param.transports = soilmap[val];
      }
      if(tag == "ERODES"){
        if(!soilmap.count(val)){
          soilmap[val] = soils.size(); //New Soil Number
          soils.push_back(param);             //Add Soil to Back
        }
        param.erodes = soilmap[val];
      }
      if(tag == "CASCADES"){
        if(!soilmap.count(val)){
          soilmap[val] = soils.size(); //New Soil Number
          soils.push_back(param);             //Add Soil to Back
        }
        param.cascades = soilmap[val];
      }
      if(tag == "ABRADES"){
        if(!soilmap.count(val)){
          soilmap[val] = soils.size(); //New Soil Number
          soils.push_back(param);             //Add Soil to Back
        }
        param.abrades = soilmap[val];
      }

      if(tag == "DENSITY")
        param.density = stof(val);
      if(tag == "POROSITY")
        param.porosity = stof(val);
      if(tag == "COLOR")
        param.color = hexcol(val);
      if(tag == "SOLUBILITY")
        param.solubility = stof(val);
      if(tag == "EQUILIBRIUM")
        param.equrate = stof(val);
      if(tag == "FRICTION")
        param.friction = stof(val);

      if(tag == "EROSIONRATE")
        param.erosionrate = stof(val);
      if(tag == "MAXDIFF")
        param.maxdiff = stof(val);
      if(tag == "SETTLING")
        param.settling = stof(val);

      if(tag == "SUSPENSION")
        param.suspension = stof(val);

      if(tag == "ABRASION")
        param.abrasion = stof(val);

      if(tag == "Ka") param.phong.x = stof(val);
      if(tag == "Kd") param.phong.y = stof(val);
      if(tag == "Ks") param.phong.z = stof(val);
      if(tag == "Kk") param.phong.w = stof(val);

    }


    if(soillayer == "LAYER"){

      if(tag == "MIN")
        layers.back().min = stof(val);
      if(tag == "BIAS")
        layers.back().bias = stof(val);
      if(tag == "SCALE")
        layers.back().scale = stof(val);

      if(tag == "OCTAVES")
        layers.back().octaves = stof(val);
      if(tag == "LACUNARITY")
        layers.back().lacunarity = stof(val);
      if(tag == "GAIN")
        layers.back().gain = stof(val);
      if(tag == "FREQUENCY")
        layers.back().frequency = stof(val);

    }

    // if(soillayer == "WORLD"){

    //   if(tag == "SIZEX")
    //     SIZEX = stoi(val);
    //   if(tag == "SIZEY")
    //     SIZEY = stoi(val);
    //   if(tag == "SCALE")
    //     SCALE = stoi(val);
    //   if(tag == "NWIND")
    //     NWIND = stoi(val);
    //   if(tag == "NWATER")
    //     NWATER = stoi(val);
    // }

  }

  in.close();

  // Set the Phong Lighting
  for(size_t i = 0; i < soils.size(); i++)
    phong.push_back(soils[i].phong);

}
