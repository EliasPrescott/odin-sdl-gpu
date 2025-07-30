# version 460

layout(location = 0) in vec4 color;
layout(location = 1) in vec2 uv;
layout(location = 2) in vec3 normal;
layout(location = 3) in vec3 frag_pos;

layout(location = 0) out vec4 frag_color;

layout(set = 2, binding = 0) uniform sampler2D tex_sampler;

layout(set = 1, binding = 0) uniform CONSTS {
    vec3 light_pos;
    vec3 view_pos;
};

void main() {
    vec4 base_color = texture(tex_sampler, uv) * color;

    vec3 lightColor = vec3(3.0, 3.0, 3.0);
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * lightColor;

    vec3 norm = normalize(normal);
    vec3 lightDir = normalize(light_pos - frag_pos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    float specularStrength = 0.5;
    vec3 viewDir = normalize(view_pos - frag_pos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    vec3 result = (ambient + diffuse + specular) * base_color.xyz;

    frag_color = vec4(result, 1.0);
    // frag_color = texture(tex_sampler, uv) * color;
    // frag_color = color;
}
