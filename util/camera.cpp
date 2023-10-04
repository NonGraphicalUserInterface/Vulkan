#define _USE_MATH_DEFINES

#include "camera.h"
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

Camera::Camera() {
	position = glm::vec3(0.0f, -2.0f, 2.0f);
	direction = glm::normalize(-position);
	up = glm::vec3(0.0f, 1.0f, 0.0f);

	pitch = 45.0f;
	yaw = 270.0f;

	updateDirection();

	// pointTo(glm::vec3(0.0f, 0.0f, 0.0f));

	std::cout << "Camera'd" << std::endl;
}

void Camera::updateDirection() {
	if (pitch > 89.0f) pitch = 89.0f;
	if (pitch < -89.0f) pitch = -89.0f;

	direction.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
	direction.y = sin(glm::radians(pitch));
	direction.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

	forward = direction;
	forward.y = 0;

	backward = -forward;

	right = glm::cross(forward, up);

	left = -right;

	// std::cout << "sad" << std::endl;
}

void Camera::updatePosition() {
	if (keys.forward) {
		position += forward * speed;
	}

	if (keys.backward) {
		position += backward * speed;
	}

	if (keys.left) {
		position += left * speed;
	}

	if (keys.right) {
		position += right * speed;
	}
}

void Camera::pointTo(glm::vec3 target) {
	float result = glm::dot(direction, target) / (glm::length(direction) * glm::length(target - position));
	float angle = acos(result);
	std::cout << result << std::endl;
	std::cout << angle << std::endl;
	glm::vec3 axis = glm::normalize(glm::cross(direction, target));

	glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), -angle, axis);

	direction = glm::vec3(rotationMatrix * glm::vec4(direction, 1.0f));
}