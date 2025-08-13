#ifndef UNIFORMS
#define UNIFORMS

struct MyUniforms {
    glm::mat4 projectionMatrix;
    glm::mat4 infiniteProjectionMatrix;
    glm::mat4 viewMatrix;
    glm::mat4 modelMatrix;

	glm::mat4 inverseProjectionMatrix;
	glm::mat4 inverseViewMatrix;

    glm::mat4 lightViewMatrix;
    glm::mat4 lightProjectionMatrix;

    glm::vec3 lightDirection;
    uint32_t transparent;

    glm::ivec3 highlightedVoxelPos;
    float time;

    glm::vec3 cameraWorldPos;
    float padding1;  // For 16-byte alignment

    glm::vec3 lightPosition;
    float padding2;  // For 16-byte alignment

	glm::vec2 screenSize;

    float padding4[2];
};

static_assert(sizeof(MyUniforms) % 16 == 0);

inline std::pair<glm::vec3, glm::vec3> getSunInfo(float time, glm::vec3 sceneCenter, float sceneRadius) {
    
    float sun_angle = time * 0.1f;
    glm::vec3 sunDirection = glm::normalize(glm::vec3(sin(sun_angle), 0.75f, cos(sun_angle)));

    if (cos(sun_angle) < 0.0f) {
        sun_angle = time * 0.1;
        sunDirection = glm::normalize(glm::vec3(sin(sun_angle), 0.75f, cos(sun_angle)));
    }

    glm::vec3 sunPosition = sceneCenter + sunDirection * sceneRadius * 4.0f;

    return { sunDirection, sunPosition };
}

inline float getSceneRadius() {
    return 1200.0f;
}

// Helper function to calculate light matrices
inline std::pair<glm::mat4, glm::mat4> calculateLightMatrices(float time, const glm::vec3& sceneCenter, float sceneRadius) {
    auto [sunDirection, sunPosition] = getSunInfo(time, sceneCenter, sceneRadius);

    glm::mat4 lightViewMatrix = glm::lookAt(
        sunPosition,
        sceneCenter,
        glm::vec3(0.0f, 0.0f, 1.0f)  // Z-up coordinate system
    );

    float orthoSize = sceneRadius * 1.2f;
    glm::mat4 lightProjectionMatrix = glm::ortho(
        -orthoSize, orthoSize,
        -orthoSize, orthoSize,
        0.01f, sceneRadius * 5.0f
    );

    return { lightViewMatrix, lightProjectionMatrix };
}



#endif