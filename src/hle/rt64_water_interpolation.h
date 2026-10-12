#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

namespace RT64::WaterInterpolation {
    // The game's staggered grid moves in camera-relative 64-unit steps. Its
    // independently rounded row/origin Z coordinates need not coincide after
    // a step, so interpolate the actual previous triangles, not an ideal grid.
    struct Sample {
        float y = 0, u = 0, v = 0;
        bool valid = true;
    };

    struct History {
        uint32_t course = 0, generation = 0;
        std::array<int32_t, 4> viewport{};
        std::array<float, 3> camera{}, forward{};
        float previousTime = 0, currentTime = 0;
        bool valid = false;
    };

    inline bool compatible(const History &cur, const History &prev) {
        if (!cur.valid || !prev.valid || cur.course != prev.course || cur.generation != prev.generation || cur.viewport != prev.viewport ||
            !std::isfinite(cur.previousTime) || !std::isfinite(cur.currentTime) || !std::isfinite(prev.currentTime) ||
            std::abs(cur.previousTime - prev.currentTime) > 0.0001f || cur.currentTime < cur.previousTime || cur.currentTime - cur.previousTime > 0.2f) {
            return false;
        }

        float distance2 = 0, facing = 0;
        for (uint32_t i = 0; i < 3; i++) {
            const float delta = cur.camera[i] - prev.camera[i];
            distance2 += delta * delta;
            facing += cur.forward[i] * prev.forward[i];
        }
        // A camera cut must not carry another view's patch into this one.
        return std::isfinite(distance2) && distance2 <= 256.0f * 256.0f && std::isfinite(facing) && facing >= 0.5f;
    }

    class PreviousMesh {
        struct Triangle {
            std::array<uint32_t, 3> indices;
            float minX, minZ, maxX, maxZ;
        };
        struct Node {
            float minX, minZ, maxX, maxZ;
            uint32_t begin, end, left = 0, right = 0;
        };
        const float *positions;
        const float *texcoords;
        std::map<std::pair<float, float>, Sample> points;
        std::vector<Triangle> triangles;
        std::vector<uint32_t> order;
        std::vector<Node> nodes;

        static bool same(const Sample &a, const Sample &b) {
            return std::abs(a.y - b.y) < 0.001f && std::abs(a.u - b.u) < 0.0001f && std::abs(a.v - b.v) < 0.0001f;
        }

        uint32_t build(uint32_t begin, uint32_t end) {
            Node node{INFINITY, INFINITY, -INFINITY, -INFINITY, begin, end};
            for (uint32_t i = begin; i < end; i++) {
                const Triangle &t = triangles[order[i]];
                node.minX = std::min(node.minX, t.minX); node.minZ = std::min(node.minZ, t.minZ);
                node.maxX = std::max(node.maxX, t.maxX); node.maxZ = std::max(node.maxZ, t.maxZ);
            }
            const uint32_t index = uint32_t(nodes.size());
            nodes.push_back(node);
            if (end - begin > 8) {
                const uint32_t middle = begin + (end - begin) / 2;
                const bool splitX = node.maxX - node.minX >= node.maxZ - node.minZ;
                std::nth_element(order.begin() + begin, order.begin() + middle, order.begin() + end, [&](uint32_t a, uint32_t b) {
                    const Triangle &ta = triangles[a], &tb = triangles[b];
                    return splitX ? ta.minX + ta.maxX < tb.minX + tb.maxX : ta.minZ + ta.maxZ < tb.minZ + tb.maxZ;
                });
                const uint32_t left = build(begin, middle), right = build(middle, end);
                nodes[index].left = left; nodes[index].right = right;
            }
            return index;
        }

        bool sampleTriangle(const Triangle &triangle, float x, float z, Sample &sample) const {
            const float *a = positions + triangle.indices[0] * 3;
            const float *b = positions + triangle.indices[1] * 3;
            const float *c = positions + triangle.indices[2] * 3;
            const double denominator = double(b[2] - c[2]) * (a[0] - c[0]) + double(c[0] - b[0]) * (a[2] - c[2]);
            if (std::abs(denominator) < 0.001) return false;
            const double wa = (double(b[2] - c[2]) * (x - c[0]) + double(c[0] - b[0]) * (z - c[2])) / denominator;
            const double wb = (double(c[2] - a[2]) * (x - c[0]) + double(a[0] - c[0]) * (z - c[2])) / denominator;
            const double wc = 1.0 - wa - wb;
            if (wa < -1e-6 || wb < -1e-6 || wc < -1e-6) return false;
            sample = {};
            const double weights[] = {wa, wb, wc};
            for (uint32_t i = 0; i < 3; i++) {
                const uint32_t vertex = triangle.indices[i];
                sample.y += float(weights[i] * positions[vertex * 3 + 1]);
                sample.u += float(weights[i] * texcoords[vertex * 2]);
                sample.v += float(weights[i] * texcoords[vertex * 2 + 1]);
            }
            return true;
        }

        bool sampleNode(uint32_t index, float x, float z, Sample &sample) const {
            const Node &node = nodes[index];
            if (x < node.minX || x > node.maxX || z < node.minZ || z > node.maxZ) return false;
            if (node.left != 0) return sampleNode(node.left, x, z, sample) || sampleNode(node.right, x, z, sample);
            for (uint32_t i = node.begin; i < node.end; i++) {
                if (sampleTriangle(triangles[order[i]], x, z, sample)) return true;
            }
            return false;
        }

    public:
        PreviousMesh(const float *positions, const float *texcoords, uint32_t vertexCount, const std::vector<uint32_t> &indices)
            : positions(positions), texcoords(texcoords) {
            for (uint32_t i = 0; i < vertexCount; i++) {
                const float *p = positions + i * 3, *tc = texcoords + i * 2;
                if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]) || !std::isfinite(tc[0]) || !std::isfinite(tc[1])) continue;
                const Sample sample{p[1], tc[0], tc[1]};
                auto result = points.emplace(std::make_pair(p[0], p[2]), sample);
                // RSP reloads produce duplicate vertices. Equal copies are
                // harmless; conflicting surfaces at one XZ cannot be matched.
                if (!result.second && !same(result.first->second, sample)) result.first->second.valid = false;
            }
            for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                Triangle triangle{{indices[i], indices[i + 1], indices[i + 2]}, INFINITY, INFINITY, -INFINITY, -INFINITY};
                bool valid = true;
                for (uint32_t vertex : triangle.indices) {
                    if (vertex >= vertexCount) { valid = false; break; }
                    const float *p = positions + vertex * 3;
                    if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2])) { valid = false; break; }
                    const auto point = points.find({p[0], p[2]});
                    if (point == points.end() || !point->second.valid) { valid = false; break; }
                    triangle.minX = std::min(triangle.minX, p[0]); triangle.minZ = std::min(triangle.minZ, p[2]);
                    triangle.maxX = std::max(triangle.maxX, p[0]); triangle.maxZ = std::max(triangle.maxZ, p[2]);
                }
                if (valid) triangles.push_back(triangle);
            }
            order.resize(triangles.size());
            std::iota(order.begin(), order.end(), 0u);
            nodes.reserve(triangles.size() * 2);
            if (!triangles.empty()) build(0, uint32_t(triangles.size()));
        }

        bool sample(float x, float z, Sample &sample) const {
            if (!std::isfinite(x) || !std::isfinite(z)) return false;
            const auto exact = points.find({x, z});
            if (exact != points.end()) {
                sample = exact->second;
                return sample.valid;
            }
            return !nodes.empty() && sampleNode(0, x, z, sample);
        }

        uint32_t velocities(const float *currentPositions, const float *currentTexcoords, uint32_t currentCount, float *positionVelocity, float *texcoordVelocity) const {
            uint32_t matched = 0;
            for (uint32_t i = 0; i < currentCount; i++) {
                positionVelocity[i * 3] = positionVelocity[i * 3 + 1] = positionVelocity[i * 3 + 2] = 0;
                texcoordVelocity[i * 2] = texcoordVelocity[i * 2 + 1] = 0;
                Sample previous;
                if (!std::isfinite(currentPositions[i * 3 + 1]) || !std::isfinite(currentTexcoords[i * 2]) || !std::isfinite(currentTexcoords[i * 2 + 1])) continue;
                if (!sample(currentPositions[i * 3], currentPositions[i * 3 + 2], previous)) continue;
                positionVelocity[i * 3 + 1] = currentPositions[i * 3 + 1] - previous.y;
                texcoordVelocity[i * 2] = currentTexcoords[i * 2] - previous.u;
                texcoordVelocity[i * 2 + 1] = currentTexcoords[i * 2 + 1] - previous.v;
                matched++;
            }
            return matched;
        }
    };
}
