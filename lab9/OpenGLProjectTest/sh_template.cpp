#define GLEW_STATIC
#include <GL/glew.h>
#include <stdio.h>
#include <string>
#include <stdlib.h>

void checkErrors(std::string desc);

const GLchar *vpSrc[] = {
		"#version 330 core\n",
		"layout(location = 0) in vec3 pos;\
         layout(location = 1) in vec3 color;\
         out vec4 vs_color;\
         void main(){\
           gl_Position = vec4(pos, 1);\
           vs_color=vec4(color, 1.0);\
           }"
};

const GLchar *fpSrc[] = {
	  "#version 330 core\n",
	  "in vec4 vs_color;\
          out vec4 fcolor;\
          void main(){\
          fcolor = vs_color;\
          }"
};

GLuint genRenderProg(){
	int rvalue;
	
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 2, vpSrc, NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &rvalue);
	if (!rvalue) {
		fprintf(stderr, "Error in compiling fp\n");
		exit(30);
	}

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 2, fpSrc, NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &rvalue);
	if (!rvalue) {
		fprintf(stderr, "Error in compiling fp\n");
		exit(31);
	}
    
	GLuint progHandle = glCreateProgram();
	glAttachShader(progHandle, vertexShader);
    glAttachShader(progHandle, fragmentShader);
    glLinkProgram(progHandle);

    glGetProgramiv(progHandle, GL_LINK_STATUS, &rvalue);
    if(!rvalue){
        fprintf(stderr, "Error in linking sp\n");
        exit(32);
    }
    checkErrors("Render shaders");
    
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

    return progHandle;
}
    
