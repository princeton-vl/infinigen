/*
COPYRIGHT


Original files authored by Nick McDonald: https://github.com/weigert/SoilMachine/tree/master


All other contributions:
Copyright (c) Princeton University.
Licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.
Authors: Zeyu Ma

*/

#define POOLSIZE 10000000

#include "include/vertexpool.h"
#include "layermap.h"
#include "particle/water.h"
#include "particle/wind.h"
#include "io.h"
#include <iostream>
using namespace std;


extern "C" {

	void run(
		float *heightmap,
		float *result_heightmap, float *result_watertrack,
		int SIZEX, int SIZEY,
		int SEED,
		int NWATER,
		int NWIND,
		float SCALE0,
		char* soil_file
	) {
		layers.clear();
        phong.clear();
        srand(1);
		Vertexpool<Vertex> vertexpool;
		loadsoil(soil_file);
		WaterParticle::init(SIZEX, SIZEY);
		WindParticle::init(SIZEX, SIZEY);
		Layermap map(SEED, glm::ivec2(SIZEX, SIZEY), SCALE0, heightmap);

		for(int i = 0; i < NWATER; i++){
			WaterParticle particle(map, SCALE0);

			while(true){
				while(particle.move(map, vertexpool) && particle.interact(map, vertexpool));
				if(!particle.flood(map, vertexpool))
					break;
			}

		}

		WaterParticle::seep(SCALE0, map, vertexpool);

		for(int i = 0; i < NWIND; i++){
			WindParticle particle(map, SCALE0);
			while(particle.move(map, vertexpool) && particle.interact(map, vertexpool));
		}

		for (int i = 0; i < SIZEX; i++)
			for (int j = 0; j < SIZEY; j++) {
				result_heightmap[i * SIZEY + j] = map.height(glm::ivec2(i, j));
			}
		
			
		for (int i = 0; i < SIZEX; i++)
			for (int j = 0; j < SIZEY; j++) {
				result_watertrack[i * SIZEY + j] = WaterParticle::track[j * SIZEX + i];
			}

		

	}
}