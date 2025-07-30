# version 460

layout(set = 1, binding = 0) uniform UBO {
    mat4 projection;
    mat4 view;
    mat4 model;
};

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec3 normal;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_uv;
layout(location = 2) out vec3 out_normal;
layout(location = 3) out vec3 out_frag_pos;

void main() {
    gl_Position = projection * view * model * vec4(position, 1);
    out_color = color;
    out_uv = uv;
    // out_normal = normal;
    out_normal = mat3(transpose(inverse(model))) * normal;
    out_frag_pos = vec3(model * vec4(position, 1.0));
}
