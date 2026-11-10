#ifndef IONMD_VEC3_HPP
#define IONMD_VEC3_HPP

/**
 * Minimal 3-vector used throughout IonMD.
 */

#include <cmath>

#if defined(__CUDACC__)
#define IONMD_HD __host__ __device__
#else
#define IONMD_HD
#endif

namespace ionmd {

struct Vec3
{
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    IONMD_HD Vec3() = default;
    IONMD_HD Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    IONMD_HD double &operator[](int i) { return i == 0 ? x : (i == 1 ? y : z); }
    IONMD_HD double operator[](int i) const { return i == 0 ? x : (i == 1 ? y : z); }

    IONMD_HD Vec3 &operator+=(const Vec3 &o) { x += o.x; y += o.y; z += o.z; return *this; }
    IONMD_HD Vec3 &operator-=(const Vec3 &o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    IONMD_HD Vec3 &operator*=(double s) { x *= s; y *= s; z *= s; return *this; }
    IONMD_HD Vec3 &operator/=(double s) { x /= s; y /= s; z /= s; return *this; }

    IONMD_HD double norm2() const { return x*x + y*y + z*z; }
    IONMD_HD double norm() const { return std::sqrt(norm2()); }

    /// Return a unit vector in the same direction (zero vector stays zero).
    IONMD_HD Vec3 normalized() const
    {
        const double n = norm();
        return n > 0.0 ? Vec3(x/n, y/n, z/n) : Vec3();
    }
};

IONMD_HD inline Vec3 operator+(Vec3 a, const Vec3 &b) { return a += b; }
IONMD_HD inline Vec3 operator-(Vec3 a, const Vec3 &b) { return a -= b; }
IONMD_HD inline Vec3 operator-(const Vec3 &a) { return Vec3(-a.x, -a.y, -a.z); }
IONMD_HD inline Vec3 operator*(Vec3 a, double s) { return a *= s; }
IONMD_HD inline Vec3 operator*(double s, Vec3 a) { return a *= s; }
IONMD_HD inline Vec3 operator/(Vec3 a, double s) { return a /= s; }

IONMD_HD inline double dot(const Vec3 &a, const Vec3 &b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

IONMD_HD inline Vec3 cross(const Vec3 &a, const Vec3 &b)
{
    return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

}  // namespace ionmd

#endif
