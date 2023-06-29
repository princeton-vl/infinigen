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
														TinyEngine Vertex Pool
================================================================================
*/

// using namespace glm;

struct Vertex {

	// Vertex(vec3 p, vec3 n, vec4 c, int i){
	// 	position[0] = p.x;
	// 	position[1] = p.y;
	// 	position[2] = p.z;
	// 	normal[0] = n.x;
	// 	normal[1] = n.y;
	// 	normal[2] = n.z;
	// 	color[0] = c.x;
	// 	color[1] = c.y;
	// 	color[2] = c.z;
	// 	color[3] = c.w;
	// 	index = i;
	// }

	// float position[3];
	// float normal[3];
	// float color[4];
	// float index;

//   static void format(int vbo){

//     glEnableVertexAttribArray(0);
//     glEnableVertexAttribArray(1);
// 		glEnableVertexAttribArray(2);
// 		glEnableVertexAttribArray(3);

//     glVertexAttribFormat(0, 3, GL_FLOAT, GL_FALSE, 0);
//     glVertexAttribFormat(1, 3, GL_FLOAT, GL_FALSE, 0);
// 		glVertexAttribFormat(2, 4, GL_FLOAT, GL_FALSE, 0);
// 		glVertexAttribFormat(3, 1, GL_FLOAT, GL_FALSE, 0);

// 		glVertexBindingDivisor(0, 0);
// 		glVertexBindingDivisor(1, 0);
// 		glVertexBindingDivisor(2, 0);
// 		glVertexBindingDivisor(3, 0);

//     glVertexAttribBinding(0, 0);
//     glVertexAttribBinding(1, 1);
// 		glVertexAttribBinding(2, 2);
// 		glVertexAttribBinding(3, 3);

//     glBindVertexBuffer(0, vbo, offsetof(Vertex, position), sizeof(Vertex));		//Internal Offset vs. Full Offset
//     glBindVertexBuffer(1, vbo, offsetof(Vertex, normal), sizeof(Vertex));
// 		glBindVertexBuffer(2, vbo, offsetof(Vertex, color), sizeof(Vertex));
// 		glBindVertexBuffer(3, vbo, offsetof(Vertex, index), sizeof(Vertex));

//   }

};

template<typename T>
class Vertexpool {

};
/*
================================================================================
                        Drawing Command Struct (Extended)
================================================================================
*/

// using namespace std;

// struct DAIC {
//   DAIC(){}
//   DAIC(uint c, uint iC, uint s, uint bV, uint* i, uint g){
//     cnt = c; instCnt = iC; start = s; baseVert = bV; baseInst = 0;
// 		index = i; group = g;
//   }

//   uint cnt;							//Base Properties
//   uint instCnt;
//   uint start;
// 	uint baseVert;
//   uint baseInst;

// 	uint* index = NULL;		//Index Pointer
// 	uint group;						//Group Assignment
// 	vec3 pos;

// };

// /*
// ================================================================================
// 	                        Vertexpool Master Class
// ================================================================================
// */

// template<typename T>
// class Vertexpool {
// private:

// GLuint vao;     //Vertex Array Object
// GLuint vbo;     //Vertex Buffer Object
// GLuint ebo;			//Element Array Buffer Object
// GLuint indbo;   //Indirect Draw Command Buffer Object

// size_t K = 0;   //Number of Vertices per Bucket
// size_t N = 0;   //Number of Maximum Buckets
// size_t M = 0;		//Number of Active Buckets
// size_t MAXSIZE = 0;

// vector<DAIC> indirect;  //Indirect Drawing Commands

// Vertexpool(){
//   glGenVertexArrays(1, &vao); //VAO Generation
//   glBindVertexArray(vao);
// 	glGenBuffers(1, &vbo);			//Buffer Generation
// 	glGenBuffers(1, &ebo);
//   glGenBuffers(1, &indbo);
//   T::format(vbo);							//Buffer Formatting
// }

// public:

// vector<GLuint> indices;

// Vertexpool(int k, int n):Vertexpool(){
//   reserve(k, n);
// }

// ~Vertexpool(){

// 	for(size_t i = 0; i < indirect.size();)
// 		unsection(indirect[i].index);

// 	glBindVertexArray(vao);
// 	glUnmapBuffer(vbo);
// 	glDeleteBuffers(1, &vbo);
// 	glDeleteBuffers(1, &ebo);
// 	glDeleteBuffers(1, &indbo);
// 	glDeleteVertexArrays(1, &vao);

// }

// /*
// ================================================================================
//                     Raw Vertex Memory Pool Management
// ================================================================================
// */

// private:

// T* start;
// deque<T*> free;

// public:

// // Allocate the Persistently Mapped Buffer, Create Buckets for Pool

// void reserve(const int k, const int n){

// 	K = k; N = n; M = n;
//   const GLbitfield flag = GL_MAP_WRITE_BIT |
// 													GL_MAP_PERSISTENT_BIT |
// 													GL_MAP_COHERENT_BIT;

//   glBindBuffer(GL_ARRAY_BUFFER, vbo);
//   glBufferStorage(GL_ARRAY_BUFFER, N*K*sizeof(T), NULL, flag);
//   start = (T*)glMapBufferRange( GL_ARRAY_BUFFER, 0, N*K*sizeof(T), flag );
// 	MAXSIZE = N*K;

// 	for(int i = 0; i < N; i++)
// 		free.push_front(start+i*K);

// }

// void clear(){

// 	while(!indirect.empty())
// 		unsection(indirect.back().index);

// 	while(!free.empty())
// 		free.pop_back();

// 	for(size_t i = 0; i < N; i++)
// 		free.push_front(start+i*K);

// 	update();

// }

// // Extract a section with a size and a group assignment

// uint* section(const int size, const int group = 0, vec3 pos = vec3(0)){

//   if(size == 0 || size > K){
// 		std::cout<<"Vertexpool Error: Insufficient Bucket Size"<<std::endl;
// 		return NULL;
// 	}
// 	if(free.empty()){
// 		std::cout<<"Vertexpool Error: No More Buckets Available"<<std::endl;
// 		return NULL;
// 	}

// 	const int first = 0;

// 	const int base = (free.back()-start);
//   indirect.emplace_back(size, 1, first, base, new uint(indirect.size()), group);
// 	free.pop_back();

// 	indirect.back().pos = pos;

// 	return indirect.back().index;

// }

// // Remove a section based on a pointer to its index

// void unsection(uint* index){

// 	if(index == NULL){
// 		std::cout<<"Vertexpool Error: Can't Unsection - Index is NULL"<<std::endl;
// 		return;
// 	}

// 	for(int k = indirect[*index].baseVert; k < K; k++)
// 		(start+k)->~T();
// 	free.push_front(start+indirect[*index].baseVert);

// 	swap(indirect[*index], indirect.back());
// 	indirect.pop_back();

// 	*indirect[*index].index = *index; //Value Copy!
// 	delete index;

// }

// // Construct a vertex in a bucket at location k on the buffer

// T* get(uint* ind, int k){
// 	return start + indirect[*ind].baseVert + k;
// }

// template<typename... Args>
// void fill(uint* ind, int k, Args && ...args){

//   T* place = get(ind, k);
// 	if(size_t(place - start) > MAXSIZE)
// 		std::cout<<"Vertexpool Error: Out-Of-Bounds Write"<<std::endl;

//   try{ new (place) T(forward<Args>(args)...); }
//   catch(...) { throw; }

// }

// /*
// ================================================================================
//             Indirect Draw Call Masking / Ordering / Manipulation
// ================================================================================
// */

// template<typename F, typename... Args>
// void mask(F function, Args&&... args){

// 	if(indirect.empty()) return;

// 	M = 0;											//Frontside Approach
// 	int J = indirect.size()-1;	//Backside Approach

// 	while(M <= J){

// 		while(function(indirect[M], args...) && M < J) M++;
// 		while(!function(indirect[J], args...) && M < J) J--;

// 		*indirect[M].index = J;
// 		*indirect[J].index = M;
// 		swap(indirect[M++], indirect[J--]);

// 	}

// }

// template<typename F, typename... Args>
// void order(F function, Args&&... args){

// 	if(indirect.empty()) return;

// 	sort(indirect.begin(), indirect.begin() + M, [&](const DAIC& a, const DAIC& b){
// 		return function(a, b, args...);
// 	});
// 	for(size_t i = 0; i < indirect.size(); i++)
// 		*indirect[i].index = i;

// }

// void resize(const uint* index, const int newsize){

// 	if(index != NULL && *index < N)
// 		indirect[*index].cnt = newsize;

// }

// /*
// ================================================================================
//           OpenGL Buffer Updating for Indexing and Indirect Calls
// ================================================================================
// */

// void index(){

// 	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
// 	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(GLuint), &indices[0], GL_STATIC_DRAW);

// }

// void update(){

//   glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indbo);
//   glBufferData(GL_DRAW_INDIRECT_BUFFER, indirect.size()*sizeof(DAIC), &indirect[0], GL_STATIC_DRAW);

// }

// /*
// ================================================================================
//                       Synchronization and Rendering
// ================================================================================
// */

// void render(const GLenum mode = GL_TRIANGLES, size_t first = 0, size_t length = 0){

// 	if(indirect.size() == 0)
// 		return;
// 	if(length > indirect.size())
// 		length = indirect.size();
// 	else if(length == 0)
// 		length = M;

//   glBindVertexArray(vao);
// 	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
//   glBindBuffer(GL_DRAW_INDIRECT_BUFFER, indbo);

// 	glMultiDrawElementsIndirect(mode, GL_UNSIGNED_INT, (void*)(first*(sizeof(DAIC))), length, sizeof(DAIC));

// }

// };
