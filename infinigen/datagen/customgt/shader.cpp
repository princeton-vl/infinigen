// Adapted from https://github.com/JoeyDeVries/LearnOpenGL/blob/master/src/7.in_practice/3.2d_game/0.full_source/shader.cpp license: CC BY 4.0

#include "shader.hpp"

// Base class

void ShaderBase::use() {
	glUseProgram(ID);
}

void ShaderBase::setFloat(const std::string& name, float value) const {
	glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
}

void ShaderBase::setDouble(const std::string& name, double value) const {
	glUniform1d(glGetUniformLocation(ID, name.c_str()), value);
}

void ShaderBase::setBool(const std::string& name, bool value) const {
	glUniform1i(glGetUniformLocation(ID, name.c_str()), int(value));
}

void ShaderBase::setInt(const std::string& name, int value) const {
	glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
}

void ShaderBase::setUInt(const std::string& name, unsigned int value) const {
	glUniform1ui(glGetUniformLocation(ID, name.c_str()), value);
}

void ShaderBase::setVec2(const std::string& name, const glm::vec2& value) const {
	glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}
void ShaderBase::setVec2(const std::string& name, float x, float y) const {
	glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y);
}

void ShaderBase::setVec3(const std::string& name, const glm::vec3& value) const {
	glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}
void ShaderBase::setVec3(const std::string& name, float x, float y, float z) const {
	glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z);
}

void ShaderBase::setVec4(const std::string& name, const glm::vec4& value) const {
	glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}
void ShaderBase::setVec4(const std::string& name, float x, float y, float z, float w) const {
	glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
}

void ShaderBase::setDVec2(const std::string& name, const glm::dvec2& value) const {
	glUniform2dv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}
void ShaderBase::setDVec2(const std::string& name, double x, double y) const {
	glUniform2d(glGetUniformLocation(ID, name.c_str()), x, y);
}

void ShaderBase::setDVec3(const std::string& name, const glm::dvec3& value) const {
	glUniform3dv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}
void ShaderBase::setDVec3(const std::string& name, double x, double y, double z) const {
	glUniform3d(glGetUniformLocation(ID, name.c_str()), x, y, z);
}

void ShaderBase::setDVec4(const std::string& name, const glm::dvec4& value) const {
	glUniform4dv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}
void ShaderBase::setDVec4(const std::string& name, double x, double y, double z, double w) const {
	glUniform4d(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
}

void ShaderBase::setIVec2(const std::string& name, const glm::ivec2& value) const {
	glUniform2iv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}
void ShaderBase::setIVec2(const std::string& name, int x, int y) const {
	glUniform2i(glGetUniformLocation(ID, name.c_str()), x, y);
}

void ShaderBase::setIVec3(const std::string& name, const glm::ivec3& value) const {
	glUniform3iv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}
void ShaderBase::setIVec3(const std::string& name, int x, int y, int z) const {
	glUniform3i(glGetUniformLocation(ID, name.c_str()), x, y, z);
}

void ShaderBase::setIVec4(const std::string& name, const glm::ivec4& value) const {
	glUniform4iv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}
void ShaderBase::setIVec4(const std::string& name, int x, int y, int z, int w) const {
	glUniform4i(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
}

void ShaderBase::setUVec2(const std::string& name, const glm::uvec2& value) const {
	glUniform2uiv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}
void ShaderBase::setUVec2(const std::string& name, unsigned int x, unsigned int y) const {
	glUniform2ui(glGetUniformLocation(ID, name.c_str()), x, y);
}

void ShaderBase::setUVec3(const std::string& name, const glm::uvec3& value) const {
	glUniform3uiv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}
void ShaderBase::setUVec3(const std::string& name, unsigned int x, unsigned int y, unsigned int z) const {
	glUniform3ui(glGetUniformLocation(ID, name.c_str()), x, y, z);
}

void ShaderBase::setUVec4(const std::string& name, const glm::uvec4& value) const {
	glUniform4uiv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
}
void ShaderBase::setUVec4(const std::string& name, unsigned int x, unsigned int y, unsigned int z, unsigned int w) const {
	glUniform4ui(glGetUniformLocation(ID, name.c_str()), x, y, z, w);
}

void ShaderBase::setMat2(const std::string& name, const glm::mat2& mat) const {
	glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void ShaderBase::setMat3(const std::string& name, const glm::mat3& mat) const {
	glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void ShaderBase::setMat4(const std::string& name, const glm::mat4& mat) const {
	glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void ShaderBase::checkCompileErrors(unsigned int shader, std::string type) {
	int success;
	char infoLog[1024];
	if (type != "PROGRAM") {
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success) {
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else {
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success) {
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}

// Child classes

Shader::Shader(const GLchar* vertexPath, const GLchar* fragmentPath, const GLchar* geometryPath) {
	std::string vertexCode, fragmentCode, geometryCode;
	std::ifstream vShaderFile, fShaderFile, gShaderFile;

	vShaderFile.open(vertexPath);
	fShaderFile.open(fragmentPath);
	gShaderFile.open(geometryPath);
	std::stringstream vShaderStream, fShaderStream, gShaderStream;

	vShaderStream << vShaderFile.rdbuf();
	fShaderStream << fShaderFile.rdbuf();
	gShaderStream << gShaderFile.rdbuf();

	vShaderFile.close();
	fShaderFile.close();
	gShaderFile.close();

	vertexCode = vShaderStream.str();
	fragmentCode = fShaderStream.str();
	geometryCode = gShaderStream.str();

	const char* vShaderCode = vertexCode.c_str();
	const char* fShaderCode = fragmentCode.c_str();
	const char* gShaderCode = geometryCode.c_str();

	// Compiling the shaders

	unsigned int vertex, fragment, geometry;

	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	checkCompileErrors(vertex, "VERTEX");

	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	checkCompileErrors(fragment, "FRAGMENT");

	geometry = glCreateShader(GL_GEOMETRY_SHADER);
	glShaderSource(geometry, 1, &gShaderCode, NULL);
	glCompileShader(geometry);
	checkCompileErrors(geometry, "GEOMETRY");

	// Shader program

	ID = glCreateProgram();
	glAttachShader(ID, vertex);
	glAttachShader(ID, fragment);
	glAttachShader(ID, geometry);
	glLinkProgram(ID);
	checkCompileErrors(ID, "PROGRAM");

	glDeleteShader(vertex);
	glDeleteShader(fragment);
	glDeleteShader(geometry);
}

Shader::Shader(const GLchar* vertexPath, const GLchar* fragmentPath) {
	std::string vertexCode, fragmentCode;
	std::ifstream vShaderFile, fShaderFile;

	vShaderFile.open(vertexPath);
	fShaderFile.open(fragmentPath);
	std::stringstream vShaderStream, fShaderStream;

	vShaderStream << vShaderFile.rdbuf();
	fShaderStream << fShaderFile.rdbuf();

	vShaderFile.close();
	fShaderFile.close();

	vertexCode = vShaderStream.str();
	fragmentCode = fShaderStream.str();

	const char* vShaderCode = vertexCode.c_str();
	const char* fShaderCode = fragmentCode.c_str();

	// Compiling the shaders

	unsigned int vertex, fragment;
        // std::cout << "#10.5\n";

	vertex = glCreateShader(GL_VERTEX_SHADER);
        // std::cout << "#10.6\n";
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	checkCompileErrors(vertex, "VERTEX");

	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	checkCompileErrors(fragment, "FRAGMENT");

	// Shader program

	ID = glCreateProgram();
	glAttachShader(ID, vertex);
	glAttachShader(ID, fragment);
	glLinkProgram(ID);
	checkCompileErrors(ID, "PROGRAM");

	glDeleteShader(vertex);
	glDeleteShader(fragment);
}

ComputeShader::ComputeShader(const GLchar* path) {
	std::string code;
	std::ifstream shaderFile;

	shaderFile.open(path);
	std::stringstream shaderStream;

	shaderStream << shaderFile.rdbuf();
	shaderFile.close();
	code = shaderStream.str();

	const char* shaderCode = code.c_str();

	// Compiling the shaders

	unsigned int shader = glCreateShader(GL_COMPUTE_SHADER);
	glShaderSource(shader, 1, &shaderCode, NULL);
	glCompileShader(shader);
	checkCompileErrors(shader, "COMPUTE");

	// Shader program

	ID = glCreateProgram();
	glAttachShader(ID, shader);
	glLinkProgram(ID);
	checkCompileErrors(ID, "PROGRAM");

	glDeleteShader(shader);
}

Shader::Shader() {
	ID = 0;
}

ComputeShader::ComputeShader() {
	ID = 0;
}
