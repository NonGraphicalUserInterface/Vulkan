#pragma once

#include <glm/glm.hpp>

struct Keys {
	bool forward;
	bool backward;
	bool right;
	bool left;
};

class Camera {
public:
	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 up;

	glm::vec3 forward;
	glm::vec3 backward;
	glm::vec3 left;
	glm::vec3 right;

	Keys keys;

	float speed = 0.01;

	double pitch, yaw;

	Camera();

	void updateDirection();
	void updatePosition();
	void pointTo(glm::vec3 target);
};