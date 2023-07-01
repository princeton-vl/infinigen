// Adapted from https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/7.in_practice/3.2d_game/0.full_source/shader.h license: CC BY 4.0

#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

//#ifndef APIENTRYP
//#define APIENTRYP APIENTRY *
//#endif

//typedef void (*PFNGLCREATESHADERPROC)(EGLenum type);
//PFNGLCREATESHADERPROC glCreateShader  = (PFNGLCREATESHADERPROC)eglGetProcAddress("glCreateShader");

class ShaderBase {
public:
	unsigned int ID;

	void use();

	void setFloat(const std::string& name, float value) const;
	void setDouble(const std::string& name, double value) const;
	void setBool(const std::string& name, bool value) const;
	void setInt(const std::string& name, int value) const;
	void setUInt(const std::string& name, unsigned int value) const;

	void setVec2(const std::string& name, const glm::vec2& value) const;
	void setVec2(const std::string& name, float x, float y) const;
	void setVec3(const std::string& name, const glm::vec3& value) const;
	void setVec3(const std::string& name, float x, float y, float z) const;
	void setVec4(const std::string& name, const glm::vec4& value) const;
	void setVec4(const std::string& name, float x, float y, float z, float w) const;

	void setDVec2(const std::string& name, const glm::dvec2& value) const;
	void setDVec2(const std::string& name, double x, double y) const;
	void setDVec3(const std::string& name, const glm::dvec3& value) const;
	void setDVec3(const std::string& name, double x, double y, double z) const;
	void setDVec4(const std::string& name, const glm::dvec4& value) const;
	void setDVec4(const std::string& name, double x, double y, double z, double w) const;

	void setIVec2(const std::string& name, const glm::ivec2& value) const;
	void setIVec2(const std::string& name, int x, int y) const;
	void setIVec3(const std::string& name, const glm::ivec3& value) const;
	void setIVec3(const std::string& name, int x, int y, int z) const;
	void setIVec4(const std::string& name, const glm::ivec4& value) const;
	void setIVec4(const std::string& name, int x, int y, int z, int w) const;

	void setUVec2(const std::string& name, const glm::uvec2& value) const;
	void setUVec2(const std::string& name, unsigned int x, unsigned int y) const;
	void setUVec3(const std::string& name, const glm::uvec3& value) const;
	void setUVec3(const std::string& name, unsigned int x, unsigned int y, unsigned int z) const;
	void setUVec4(const std::string& name, const glm::uvec4& value) const;
	void setUVec4(const std::string& name, unsigned int x, unsigned int y, unsigned int z, unsigned int w) const;

	void setMat2(const std::string& name, const glm::mat2& mat) const;
	void setMat3(const std::string& name, const glm::mat3& mat) const;
	void setMat4(const std::string& name, const glm::mat4& mat) const;

protected:
	void checkCompileErrors(unsigned int shader, std::string type);
};

class Shader : public ShaderBase {
public:
	Shader(const GLchar* vertexPath, const GLchar* fragmentPath, const GLchar* geometryPath);
	Shader(const GLchar* vertexPath, const GLchar* fragmentPath);
	Shader();
};

class ComputeShader : public ShaderBase {
public:
	ComputeShader(const GLchar* path);
	ComputeShader();
};
