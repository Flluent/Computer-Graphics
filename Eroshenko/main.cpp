#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <string>

using namespace std;

// Структура для вершины
struct Vertex {
    float x, y, z;
};

// Структура для нормали
struct Normal {
    float x, y, z;
};

// Структура для текстуры
struct TexCoord {
    float u, v;
};

// Структура для объекта
struct Object {
    vector<Vertex> vertices;
    vector<Normal> normals;
    vector<TexCoord> texCoords;
    vector<vector<int>> faces;
    int shadingModel; // 0 - Phong, 1 - Toon, 2 - Minnaert
    float color[3]; // Цвет объекта
    float position[3]; // Позиция объекта
    float scale;
};

// Структура для источника света
struct LightSource {
    int type; // 0 - точечный, 1 - направленный, 2 - прожектор
    float position[3];
    float direction[3];
    float intensity;
    float cutoff; // Угол конуса для прожектора
    float color[3];
};

// Глобальные переменные
vector<Object> objects;
vector<LightSource> lights;
float cameraAngle = 0.0f;
float cameraDistance = 10.0f;
bool showNormals = false;
int selectedObject = 0;
int selectedLight = 0;

// Параметры Minnaert
float minnaertK = 0.5f;

// Загрузка объекта из файла OBJ
bool loadOBJ(const string& filename, Object& obj) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Не могу открыть файл: " << filename << endl;
        return false;
    }
    
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        string type;
        iss >> type;
        
        if (type == "v") { // Вершина
            Vertex v;
            iss >> v.x >> v.y >> v.z;
            obj.vertices.push_back(v);
        }
        else if (type == "vn") { // Нормаль
            Normal n;
            iss >> n.x >> n.y >> n.z;
            obj.normals.push_back(n);
        }
        else if (type == "vt") { // Текстура
            TexCoord tc;
            iss >> tc.u >> tc.v;
            obj.texCoords.push_back(tc);
        }
        else if (type == "f") { // Полигон
            vector<int> face;
            string token;
            while (iss >> token) {
                size_t pos1 = token.find('/');
                size_t pos2 = token.find('/', pos1 + 1);
                
                int vIdx = stoi(token.substr(0, pos1)) - 1;
                int tIdx = -1;
                int nIdx = -1;
                
                if (pos2 != string::npos) {
                    if (pos2 > pos1 + 1) {
                        tIdx = stoi(token.substr(pos1 + 1, pos2 - pos1 - 1)) - 1;
                    }
                    nIdx = stoi(token.substr(pos2 + 1)) - 1;
                }
                else if (pos1 != string::npos) {
                    nIdx = stoi(token.substr(pos1 + 1)) - 1;
                }
                
                face.push_back(vIdx);
                face.push_back(nIdx);
                face.push_back(tIdx);
            }
            obj.faces.push_back(face);
        }
    }
    
    file.close();
    cout << "Загружен объект: " << filename 
         << " (вершин: " << obj.vertices.size() 
         << ", граней: " << obj.faces.size() << ")" << endl;
    return true;
}

// Функция для инициализации сцены
void initScene() {
    objects.clear();
    
    // Создаем 5 объектов с разными моделями освещения
    Object obj;
    
    // Куб (Phong)
    obj = Object();
    loadOBJ("cube.obj", obj);
    obj.shadingModel = 0; // Phong
    obj.color[0] = 0.8f; obj.color[1] = 0.2f; obj.color[2] = 0.2f;
    obj.position[0] = -3.0f; obj.position[1] = 0.0f; obj.position[2] = 0.0f;
    obj.scale = 0.5f;
    objects.push_back(obj);
    
    // Сфера (Toon)
    obj = Object();
    loadOBJ("sphere.obj", obj);
    obj.shadingModel = 1; // Toon
    obj.color[0] = 0.2f; obj.color[1] = 0.8f; obj.color[2] = 0.2f;
    obj.position[0] = -1.5f; obj.position[1] = 0.0f; obj.position[2] = 0.0f;
    obj.scale = 0.5f;
    objects.push_back(obj);
    
    // Тор (Minnaert)
    obj = Object();
    loadOBJ("torus.obj", obj);
    obj.shadingModel = 2; // Minnaert
    obj.color[0] = 0.2f; obj.color[1] = 0.2f; obj.color[2] = 0.8f;
    obj.position[0] = 0.0f; obj.position[1] = 0.0f; obj.position[2] = 0.0f;
    obj.scale = 0.5f;
    objects.push_back(obj);
    
    // Конус (Phong)
    obj = Object();
    loadOBJ("cone.obj", obj);
    obj.shadingModel = 0; // Phong
    obj.color[0] = 0.8f; obj.color[1] = 0.8f; obj.color[2] = 0.2f;
    obj.position[0] = 1.5f; obj.position[1] = 0.0f; obj.position[2] = 0.0f;
    obj.scale = 0.5f;
    objects.push_back(obj);
    
    // Цилиндр (Toon)
    obj = Object();
    loadOBJ("cylinder.obj", obj);
    obj.shadingModel = 1; // Toon
    obj.color[0] = 0.8f; obj.color[1] = 0.2f; obj.color[2] = 0.8f;
    obj.position[0] = 3.0f; obj.position[1] = 0.0f; obj.position[2] = 0.0f;
    obj.scale = 0.5f;
    objects.push_back(obj);
    
    // Создаем источники света
    LightSource light;
    
    // Точечный источник (лампочка)
    light.type = 0;
    light.position[0] = 2.0f; light.position[1] = 5.0f; light.position[2] = 2.0f;
    light.direction[0] = 0.0f; light.direction[1] = -1.0f; light.direction[2] = 0.0f;
    light.intensity = 1.0f;
    light.cutoff = 45.0f;
    light.color[0] = 1.0f; light.color[1] = 1.0f; light.color[2] = 1.0f;
    lights.push_back(light);
    
    // Направленный источник (солнце)
    light.type = 1;
    light.position[0] = 0.0f; light.position[1] = 0.0f; light.position[2] = 0.0f;
    light.direction[0] = -0.5f; light.direction[1] = -1.0f; light.direction[2] = -0.5f;
    light.intensity = 0.7f;
    light.cutoff = 180.0f;
    light.color[0] = 1.0f; light.color[1] = 1.0f; light.color[2] = 0.9f;
    lights.push_back(light);
    
    // Прожектор (фонарик)
    light.type = 2;
    light.position[0] = -2.0f; light.position[1] = 3.0f; light.position[2] = -2.0f;
    light.direction[0] = 0.5f; light.direction[1] = -0.5f; light.direction[2] = 0.5f;
    light.intensity = 0.8f;
    light.cutoff = 30.0f;
    light.color[0] = 0.9f; light.color[1] = 0.9f; light.color[2] = 1.0f;
    lights.push_back(light);
}

// Нормализация вектора
void normalize(float v[3]) {
    float length = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    if (length > 0.0f) {
        v[0] /= length;
        v[1] /= length;
        v[2] /= length;
    }
}

// Скалярное произведение
float dotProduct(float v1[3], float v2[3]) {
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
}

// Модель освещения Phong
float phongShading(float normal[3], float lightDir[3], float viewDir[3]) {
    // Диффузная составляющая
    float diff = max(dotProduct(normal, lightDir), 0.0f);
    
    // Зеркальная составляющая
    float reflectDir[3] = {
        2 * diff * normal[0] - lightDir[0],
        2 * diff * normal[1] - lightDir[1],
        2 * diff * normal[2] - lightDir[2]
    };
    normalize(reflectDir);
    float spec = pow(max(dotProduct(viewDir, reflectDir), 0.0f), 32.0f);
    
    return 0.7f * diff + 0.3f * spec;
}

// Toon Shading
float toonShading(float normal[3], float lightDir[3]) {
    float dot = dotProduct(normal, lightDir);
    
    // Квантование
    if (dot > 0.95) return 1.0f;
    else if (dot > 0.5) return 0.7f;
    else if (dot > 0.25) return 0.4f;
    else if (dot > 0.05) return 0.2f;
    else return 0.1f;
}

// Модель Minnaert
float minnaertShading(float normal[3], float lightDir[3], float viewDir[3]) {
    float NdotL = max(dotProduct(normal, lightDir), 0.0f);
    float NdotV = max(dotProduct(normal, viewDir), 0.0f);
    
    // Формула Minnaert
    return NdotL * pow(NdotL * NdotV, minnaertK - 1.0f);
}

// Расчет освещения для точки
void calculateLighting(float point[3], float normal[3], float color[3], int shadingModel) {
    float total[3] = {0.1f, 0.1f, 0.1f}; // Фоновое освещение
    
    float viewDir[3] = {0.0f, 0.0f, -1.0f}; // Направление на камеру
    normalize(viewDir);
    
    for (const auto& light : lights) {
        float lightDir[3];
        float attenuation = 1.0f;
        
        if (light.type == 0) { // Точечный источник
            lightDir[0] = light.position[0] - point[0];
            lightDir[1] = light.position[1] - point[1];
            lightDir[2] = light.position[2] - point[2];
            normalize(lightDir);
            
            // Затухание
            float distance = sqrt(
                pow(light.position[0] - point[0], 2) +
                pow(light.position[1] - point[1], 2) +
                pow(light.position[2] - point[2], 2)
            );
            attenuation = 1.0f / (1.0f + 0.1f * distance + 0.01f * distance * distance);
        }
        else if (light.type == 1) { // Направленный источник
            lightDir[0] = -light.direction[0];
            lightDir[1] = -light.direction[1];
            lightDir[2] = -light.direction[2];
            normalize(lightDir);
        }
        else if (light.type == 2) { // Прожектор
            lightDir[0] = light.position[0] - point[0];
            lightDir[1] = light.position[1] - point[1];
            lightDir[2] = light.position[2] - point[2];
            normalize(lightDir);
            
            // Проверка угла конуса
            float spotDir[3] = {light.direction[0], light.direction[1], light.direction[2]};
            normalize(spotDir);
            float cosAngle = dotProduct(lightDir, spotDir);
            float cutoffRad = cos(light.cutoff * M_PI / 180.0f);
            
            if (cosAngle < cutoffRad) {
                continue; // Точка вне конуса
            }
            
            // Затухание
            float distance = sqrt(
                pow(light.position[0] - point[0], 2) +
                pow(light.position[1] - point[1], 2) +
                pow(light.position[2] - point[2], 2)
            );
            attenuation = 1.0f / (1.0f + 0.1f * distance);
            attenuation *= pow(cosAngle, 2.0f); // Дополнительное затухание от центра конуса
        }
        
        float shade = 0.0f;
        switch(shadingModel) {
            case 0: // Phong
                shade = phongShading(normal, lightDir, viewDir);
                break;
            case 1: // Toon
                shade = toonShading(normal, lightDir);
                break;
            case 2: // Minnaert
                shade = minnaertShading(normal, lightDir, viewDir);
                break;
        }
        
        total[0] += light.color[0] * light.intensity * attenuation * shade;
        total[1] += light.color[1] * light.intensity * attenuation * shade;
        total[2] += light.color[2] * light.intensity * attenuation * shade;
    }
    
    // Ограничиваем значения
    color[0] = min(max(total[0], 0.0f), 1.0f);
    color[1] = min(max(total[1], 0.0f), 1.0f);
    color[2] = min(max(total[2], 0.0f), 1.0f);
}

// Отрисовка объекта
void drawObject(const Object& obj) {
    glPushMatrix();
    glTranslatef(obj.position[0], obj.position[1], obj.position[2]);
    glScalef(obj.scale, obj.scale, obj.scale);
    
    glBegin(GL_TRIANGLES);
    for (const auto& face : obj.faces) {
        for (size_t i = 0; i < face.size(); i += 3) {
            int vIdx = face[i];
            int nIdx = face[i+1];
            int tIdx = face[i+2];
            
            if (vIdx >= 0 && vIdx < obj.vertices.size()) {
                const Vertex& v = obj.vertices[vIdx];
                
                float normal[3];
                if (nIdx >= 0 && nIdx < obj.normals.size()) {
                    normal[0] = obj.normals[nIdx].x;
                    normal[1] = obj.normals[nIdx].y;
                    normal[2] = obj.normals[nIdx].z;
                } else {
                    // Вычисляем нормаль по вершинам
                    normal[0] = v.x; normal[1] = v.y; normal[2] = v.z;
                    normalize(normal);
                }
                
                float point[3] = {v.x, v.y, v.z};
                float color[3];
                calculateLighting(point, normal, color, obj.shadingModel);
                
                // Умножаем на цвет объекта
                color[0] *= obj.color[0];
                color[1] *= obj.color[1];
                color[2] *= obj.color[2];
                
                glColor3f(color[0], color[1], color[2]);
                glVertex3f(v.x, v.y, v.z);
            }
        }
    }
    glEnd();
    
    // Отрисовка нормалей (если включено)
    if (showNormals) {
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_LINES);
        for (const auto& face : obj.faces) {
            for (size_t i = 0; i < face.size(); i += 3) {
                int vIdx = face[i];
                int nIdx = face[i+1];
                
                if (vIdx >= 0 && vIdx < obj.vertices.size() &&
                    nIdx >= 0 && nIdx < obj.normals.size()) {
                    const Vertex& v = obj.vertices[vIdx];
                    const Normal& n = obj.normals[nIdx];
                    
                    glVertex3f(v.x, v.y, v.z);
                    glVertex3f(v.x + n.x * 0.2f, v.y + n.y * 0.2f, v.z + n.z * 0.2f);
                }
            }
        }
        glEnd();
    }
    
    glPopMatrix();
}

// Отрисовка источников света
void drawLights() {
    for (size_t i = 0; i < lights.size(); i++) {
        const auto& light = lights[i];
        
        glPushMatrix();
        glTranslatef(light.position[0], light.position[1], light.position[2]);
        
        if (i == selectedLight) {
            glColor3f(1.0f, 1.0f, 0.0f);
        } else {
            glColor3f(light.color[0], light.color[1], light.color[2]);
        }
        
        glutSolidSphere(0.2f, 10, 10);
        
        // Отрисовка направления для прожектора
        if (light.type == 2) {
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(light.direction[0] * 2, 
                      light.direction[1] * 2, 
                      light.direction[2] * 2);
            glEnd();
        }
        
        glPopMatrix();
    }
}

// Функция отрисовки
void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, 1.0f, 0.1f, 100.0f);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    
    // Позиция камеры
    float camX = cameraDistance * sin(cameraAngle * M_PI / 180.0f);
    float camZ = cameraDistance * cos(cameraAngle * M_PI / 180.0f);
    gluLookAt(camX, 3.0f, camZ, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    
    // Отрисовка объектов
    for (size_t i = 0; i < objects.size(); i++) {
        if (i == selectedObject) {
            glLineWidth(3.0f);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glColor3f(1.0f, 1.0f, 0.0f);
            glPushMatrix();
            glTranslatef(objects[i].position[0], objects[i].position[1], objects[i].position[2]);
            glScalef(objects[i].scale, objects[i].scale, objects[i].scale);
            glutWireCube(2.0f);
            glPopMatrix();
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glLineWidth(1.0f);
        }
        drawObject(objects[i]);
    }
    
    // Отрисовка источников света
    drawLights();
    
    // Отрисовка текста
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, 800, 0, 600);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    
    glColor3f(1.0f, 1.0f, 1.0f);
    glRasterPos2f(10, 580);
    string info = "Управление: стрелки - камера, 1-5 - выбор объекта, 6-8 - выбор света, N - нормали, +/- - параметры";
    for (char c : info) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);
    }
    
    glRasterPos2f(10, 560);
    string objInfo = "Объект " + to_string(selectedObject+1) + ": " + 
                     (objects[selectedObject].shadingModel == 0 ? "Phong" : 
                      objects[selectedObject].shadingModel == 1 ? "Toon" : "Minnaert");
    for (char c : objInfo) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);
    }
    
    glRasterPos2f(10, 540);
    string lightInfo = "Свет " + to_string(selectedLight+1) + ": " + 
                      (lights[selectedLight].type == 0 ? "Точечный" : 
                       lights[selectedLight].type == 1 ? "Направленный" : "Прожектор");
    for (char c : lightInfo) {
        glutBitmapCharacter(GLUT_BITMAP_9_BY_15, c);
    }
    
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    
    glutSwapBuffers();
}

// Обработка клавиатуры
void keyboard(unsigned char key, int x, int y) {
    switch(key) {
        case 27: // ESC
            exit(0);
            break;
        case 'n':
        case 'N':
            showNormals = !showNormals;
            break;
        case '1':
            selectedObject = 0;
            break;
        case '2':
            selectedObject = 1;
            break;
        case '3':
            selectedObject = 2;
            break;
        case '4':
            selectedObject = 3;
            break;
        case '5':
            selectedObject = 4;
            break;
        case '6':
            selectedLight = 0;
            break;
        case '7':
            selectedLight = 1;
            break;
        case '8':
            selectedLight = 2;
            break;
        case '+':
            if (selectedObject == 2) { // Для Minnaert
                minnaertK += 0.1f;
            } else {
                lights[selectedLight].intensity += 0.1f;
            }
            break;
        case '-':
            if (selectedObject == 2) { // Для Minnaert
                minnaertK -= 0.1f;
                if (minnaertK < 0.1f) minnaertK = 0.1f;
            } else {
                lights[selectedLight].intensity -= 0.1f;
                if (lights[selectedLight].intensity < 0.0f) 
                    lights[selectedLight].intensity = 0.0f;
            }
            break;
    }
    glutPostRedisplay();
}

// Обработка специальных клавиш
void specialKeys(int key, int x, int y) {
    switch(key) {
        case GLUT_KEY_LEFT:
            cameraAngle -= 5.0f;
            break;
        case GLUT_KEY_RIGHT:
            cameraAngle += 5.0f;
            break;
        case GLUT_KEY_UP:
            cameraDistance -= 0.5f;
            if (cameraDistance < 3.0f) cameraDistance = 3.0f;
            break;
        case GLUT_KEY_DOWN:
            cameraDistance += 0.5f;
            break;
        case GLUT_KEY_F1: // Регулировка положения света
            lights[selectedLight].position[0] += 0.5f;
            break;
        case GLUT_KEY_F2:
            lights[selectedLight].position[0] -= 0.5f;
            break;
        case GLUT_KEY_F3:
            lights[selectedLight].position[1] += 0.5f;
            break;
        case GLUT_KEY_F4:
            lights[selectedLight].position[1] -= 0.5f;
            break;
        case GLUT_KEY_F5:
            lights[selectedLight].position[2] += 0.5f;
            break;
        case GLUT_KEY_F6:
            lights[selectedLight].position[2] -= 0.5f;
            break;
    }
    glutPostRedisplay();
}

// Функция инициализации
void init() {
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_NORMALIZE);
    
    initScene();
}

// Основная функция
int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(800, 600);
    glutCreateWindow("Освещение - 3D Сцена");
    
    init();
    
    glutDisplayFunc(display);
    glutKeyboardFunc(keyboard);
    glutSpecialFunc(specialKeys);
    
    cout << "=== Управление ===" << endl;
    cout << "Стрелки: вращение камеры" << endl;
    cout << "1-5: выбор объекта" << endl;
    cout << "6-8: выбор источника света" << endl;
    cout << "N: показать/скрыть нормали" << endl;
    cout << "+/-: изменить интенсивность/параметр" << endl;
    cout << "F1-F6: перемещение выбранного источника" << endl;
    cout << "ESC: выход" << endl;
    
    glutMainLoop();
    return 0;
}