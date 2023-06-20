#pragma once
# include <glfw3.h>

class Applications
{
    GLFWwindow* window = nullptr;
public:
    Applications() {}
    bool initWindow() {
        if (!glfwInit()) return false;
        window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);
        if (!window) {
            glfwTerminate();
            return false;
        }
        glfwMakeContextCurrent(window);
        while (!glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT);
            glBegin(GL_TRIANGLES);
            glVertex2f(-0.5f, -0.5f);
            glVertex2f(0.0f, 0.5f);
            glVertex2f(0.5f, -0.5f);
            glEnd();
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
        glfwTerminate();
        return true;
    }
};

