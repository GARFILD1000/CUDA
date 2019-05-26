
#include <iostream>

// GLEW
#define GLEW_STATIC
#include <GL/glew.h>

// GLFW
#include <GLFW/glfw3.h>

// Window dimensions
const GLuint window_width = 800, window_height = 600;

// Shaders
const GLchar* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 position;\n"
"void main()\n"
"{\n"
"gl_Position = vec4(position.x, position.y, position.z, 1.0);\n"
"}\0";
const GLchar* fragmentShaderSource = "#version 330 core\n"
"out vec4 color;\n"
"void main()\n"
"{\n"
"color = vec4(0.0f, 0.5f, 0.7f, 0.5f);\n"
"}\n\0";

const GLchar *vpSrc =
"#version 430\n"
"layout(location = 0) in vec3 pos;\n"
"layout(location = 1) in vec3 color;\n"
"out vec4 vs_color;\n"
"void main(){\n"
"gl_Position = vec4(pos, 1);\n"
"vs_color=vec4(color, 1.0);\n"
"}\0";

const GLchar *fpSrc =
"#version 430\n"
"in vec4 vs_color;\n"
"out vec4 fcolor;\n"
"void main(){\n"
"fcolor = vs_color;\n"
"}\n\0";

static GLfloat vertex_buffer_data[] = {
		   -0.5f, -0.5, -0.0f, 1.0f, 0.0f, 0.0f,
		   0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f,
		   0.0f, 0.8f, 0.0f, 0.0f, 0.0f, 1.0f
};

GLFWwindow *window;
GLuint shaderProgram;
GLuint VBO, VAO;

int initBuffer() {
	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);

	
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data), vertex_buffer_data, GL_STATIC_DRAW);
	//glBufferData( GL_ARRAY_BUFFER, 6*num_of_verticles*sizeof(float),
	//              vertex_buffer_data, GL_STATIC_DRAW );
	glVertexAttribPointer(0, 6, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	return 0;
}

void initGL() {
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW!\n");
		getchar();
		return;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// Create a GLFWwindow object that we can use for GLFW's functions
	window = glfwCreateWindow(window_width, window_height, "LearnOpenGL", nullptr, nullptr);

	int screenWidth, screenHeight;
	glfwGetFramebufferSize(window, &screenWidth, &screenHeight);

	if (window == NULL) {
		fprintf(stderr, "Failed to open GLFW window. \n");
		getchar();
		glfwTerminate();
		return;
	}

	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to init GLEW\n");
		getchar();
		glfwTerminate();
		return;
	}

	glViewport(0, 0, screenWidth, screenHeight);
	return;
}

void initShaderProgram() {

	// Build and compile our shader program
	// Vertex shader
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vpSrc, NULL);
	glCompileShader(vertexShader);

	// Check for compile time errors
	GLint success;
	GLchar infoLog[512];

	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	// Fragment shader
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fpSrc, NULL);
	glCompileShader(fragmentShader);

	// Check for compile time errors
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	// Link shaders
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	// Check for linking errors
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);

	if (!success)
	{
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
	}

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
}


// The MAIN function, from here we start the application and run the game loop
int main()
{
	initGL();
	initShaderProgram();
	initBuffer();
	bool keyPressed = false;
	float moveSpeed = 0.001f;

	while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window))
	{


		// Check if any events have been activiated (key pressed, mouse moved etc.) and call corresponding response functions
		glfwPollEvents();

		if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
			vertex_buffer_data[1] -= moveSpeed;
			vertex_buffer_data[7] -= moveSpeed;
			vertex_buffer_data[13] -= moveSpeed;
			initBuffer();
			keyPressed = true;
		}
		
		if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
			vertex_buffer_data[1] += moveSpeed;
			vertex_buffer_data[7] += moveSpeed;
			vertex_buffer_data[13] += moveSpeed;
			initBuffer();
			keyPressed = true;
		}

		if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
			vertex_buffer_data[0] += moveSpeed;
			vertex_buffer_data[6] += moveSpeed;
			vertex_buffer_data[12] += moveSpeed;
			initBuffer();
			keyPressed = true;
		}

		if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
			vertex_buffer_data[0] -= moveSpeed;
			vertex_buffer_data[6] -= moveSpeed;
			vertex_buffer_data[12] -= moveSpeed;
			initBuffer();
			keyPressed = true;
		}

		if (keyPressed) {
			for (int key = 0; key < 349; key++) {
				if (glfwGetKey(window, key) == GLFW_RELEASE) {
					keyPressed = false;
					break;
				}
			}
		}

		// Render
		// Clear the colorbuffer
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glPointSize(6);

		// Draw our first triangle
		glUseProgram(shaderProgram);

		GLint posPtr = glGetAttribLocation(shaderProgram, "pos");
		glVertexAttribPointer(posPtr, 3, GL_FLOAT, GL_FALSE, 24, 0);
		glEnableVertexAttribArray(posPtr);

		GLint colorPtr = glGetAttribLocation(shaderProgram, "color");
		glVertexAttribPointer(colorPtr, 3, GL_FLOAT, GL_FALSE, 24, (const GLvoid*)12);
		glEnableVertexAttribArray(colorPtr);


		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, 3);
		glBindVertexArray(0);


		// Swap the screen buffers
		glfwSwapBuffers(window);
	}

	// Properly de-allocate all resources once they've outlived their purpose
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);

	// Terminate GLFW, clearing any resources allocated by GLFW.
	glfwTerminate();

	return EXIT_SUCCESS;
}

void display(GLuint progHandle) {
	glUseProgram(progHandle);

	GLint posPtr = glGetAttribLocation(progHandle, "pos");
	glVertexAttribPointer(posPtr, 3, GL_FLOAT, GL_FALSE, 24, 0);
	glEnableVertexAttribArray(posPtr);

	GLint colorPtr = glGetAttribLocation(progHandle, "color");
	glVertexAttribPointer(colorPtr, 3, GL_FLOAT, GL_FALSE, 24, (const GLvoid*)12);

	glEnableVertexAttribArray(colorPtr);

	glDrawArrays(GL_TRIANGLES, 0, 3);

	glDisableVertexAttribArray(posPtr);
	glDisableVertexAttribArray(colorPtr);
}


/*
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>


#include <stdio.h>
#include <string>
#include <stdlib.h>

void initGL();
int initBuffer();
void display(GLuint progHandle);
void myCleanup(GLuint progHandle);
GLuint genRenderProg();

GLFWwindow *window;

const unsigned int window_width = 512;
const unsigned int window_height = 512;

int main(){
    initGL();

	GLuint progHandle = genRenderProg();

	initBuffer();

	printf("All configured, starting main 'while'...\n");
    while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS &&
		glfwWindowShouldClose(window) == 0) {
		printf("'While'\n");

		glfwPollEvents();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glPointSize(6);
        display(progHandle);
        glfwSwapBuffers(window);      
	};
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
    myCleanup(progHandle);
    glfwTerminate();
    return 0;
}

void initGL(){
    if (!glfwInit()){
        fprintf(stderr, "Failed to initialize GLFW!\n");
        getchar();
        return;                
    }   

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	// Create a GLFWwindow object that we can use for GLFW's functions
	GLFWwindow *window = glfwCreateWindow(window_width, window_height, "LearnOpenGL", nullptr, nullptr);

	int screenWidth, screenHeight;
	glfwGetFramebufferSize(window, &screenWidth, &screenHeight);
    
    if( window==NULL ){
        fprintf(stderr,"Failed to open GLFW window. \n");
        getchar();
        glfwTerminate();
        return;    
    }

    glfwMakeContextCurrent(window);
    
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK){
        fprintf(stderr, "Failed to init GLEW\n");               
        getchar();
        glfwTerminate();
        return;           
    }

	glViewport(0, 0, screenWidth, screenHeight);
    return;
}


void checkErrors(std::string desc) {
	GLenum e = glGetError();
	if (e != GL_NO_ERROR) {
		fprintf(stderr, "OpenGL error in %s: %s (%d)\n", desc.c_str(), "Glu error string", e);
		exit(20);
	}
}

GLuint bufferID;
GLuint vertexArrays;
GLuint genRenderProg();
const int num_of_verticles = 3;

int initBuffer() {
	glGenVertexArrays(1, &vertexArrays);
	glGenBuffers(1, &bufferID);

	glBindVertexArray(vertexArrays);
	glBindBuffer(GL_ARRAY_BUFFER, bufferID);

	static const GLfloat vertex_buffer_data[] = {
		   -0.9f, -0.9, -0.0f, 1.0f, 0.0f, 0.0f,
		   0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
		   0.9f, -0.5f, 0.0f, 0.0f, 0.0f, 1.0f
	};
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_buffer_data), vertex_buffer_data, GL_STATIC_DRAW);
	//glBufferData( GL_ARRAY_BUFFER, 6*num_of_verticles*sizeof(float),
	//              vertex_buffer_data, GL_STATIC_DRAW );
	glVertexAttribPointer(0, 6, GL_FLOAT, GL_FALSE, 6 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	return 0;
}

void display(GLuint progHandle) {
	glUseProgram(progHandle);

	GLint posPtr = glGetAttribLocation(progHandle, "pos");
	glVertexAttribPointer(posPtr, 3, GL_FLOAT, GL_FALSE, 24, 0);
	glEnableVertexAttribArray(posPtr);

	GLint colorPtr = glGetAttribLocation(progHandle, "color");
	glVertexAttribPointer(colorPtr, 3, GL_FLOAT, GL_FALSE, 24, (const GLvoid*)12);

	glEnableVertexAttribArray(colorPtr);

	glDrawArrays(GL_TRIANGLES, 0, num_of_verticles);

	glDisableVertexAttribArray(posPtr);
	glDisableVertexAttribArray(colorPtr);
}

void myCleanup(GLuint progHandle) {
	glDeleteBuffers(1, &bufferID);
	glDeleteProgram(progHandle);
}


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

GLuint genRenderProg() {
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
	if (!rvalue) {
		fprintf(stderr, "Error in linking sp\n");
		exit(32);
	}
	checkErrors("Render shaders");

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return progHandle;
}


*/