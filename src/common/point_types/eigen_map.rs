#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]

//! EIGEN_MAP & PCL_ADD线性代数计算适配, 内存对齐

use nalgebra::{ArrayStorage, Const, Matrix, Scalar, Vector2, Vector3, Vector4, U1, U3, U4};
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

/* start 线性代数适配nalgebra库 */

// 定义类型别名，用于映射不同类型的向量
pub type Vector2fMap = Vector2<f32>;
pub type Vector2fMapConst = Vector2<f32>;
pub type Array3fMap = Vector3<f32>;
pub type Array3fMapConst = Vector3<f32>;
pub type Array4fMap = Vector4<f32>;
pub type Array4fMapConst = Vector4<f32>;
pub type Vector3fMap = Vector3<f32>;
pub type Vector3fMapConst = Vector3<f32>;
pub type Vector4fMap = Vector4<f32>;
pub type Vector4fMapConst = Vector4<f32>;

pub type Vector3c = Vector3<u8>;
pub type Vector3cMap = Vector3c;
pub type Vector3cMapConst = Vector3c;
pub type Vector4c = Vector4<u8>;
pub type Vector4cMap = Vector4c;
pub type Vector4cMapConst = Vector4c;

// 1. 定义Point4D结构体，包含x,y,z坐标
// 原来的PCL_ADD_UNION_POINT4D, 不用union这种不安全方式
#[derive(Debug, Clone, Copy)]
pub struct PclAddPoint4D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub data: [f32; 4],
}

// 原来的PCL_ADD_EIGEN_MAPS_POINT4D
impl PclAddPoint4D {
    // 获取2D向量映射
    pub fn get_vector2f_map(&self) -> Vector2fMap {
        Vector2fMap::new(self.data[0], self.data[1])
    }

    // 获取3D向量映射
    pub fn get_vector3f_map(&self) -> Vector3fMap {
        Vector3fMap::new(self.data[0], self.data[1], self.data[2])
    }

    // 获取4D向量映射
    pub fn get_vector4f_map(&self) -> Vector4fMap {
        Vector4fMap::new(self.data[0], self.data[1], self.data[2], self.data[3])
    }
}

// 2. 定义Normal4D结构体，包含法线向量
// 原来的PCL_ADD_UNION_NORMAL4D, 不用union这种不安全方式
#[derive(Debug, Clone, Copy)]
pub struct PclAddNormal4D {
    pub normal_x: f32,
    pub normal_y: f32,
    pub normal_z: f32,
    pub data_n: [f32; 4],
    pub normal: [f32; 3],
}

// 原来的PCL_ADD_EIGEN_MAPS_NORMAL4D
impl PclAddNormal4D {
    // 获取3D法线向量映射
    pub fn get_normal_vector3f_map(&self) -> Vector3fMap {
        Vector3fMap::new(self.data_n[0], self.data_n[1], self.data_n[2])
    }

    // 获取4D法线向量映射
    pub fn get_normal_vector4f_map(&self) -> Vector4fMap {
        Vector4fMap::new(
            self.data_n[0],
            self.data_n[1],
            self.data_n[2],
            self.data_n[3],
        )
    }
}

// 3. 定义RGB颜色结构体
// 原来的PCL_ADD_UNION_RGB, 不用union这种不安全方式
#[derive(Debug, Clone, Copy)]
pub struct PclAddRGB {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
    pub rgba: u32,
}

impl PclAddRGB {
    // 获取RGB 3D向量
    pub fn get_rgb_vector3i(&self) -> Vector3<u8> {
        Vector3::new(self.r, self.g, self.b)
    }

    // 获取RGBA 4D向量
    pub fn get_rgba_vector4i(&self) -> Vector4<u8> {
        Vector4::new(self.r, self.g, self.b, self.a)
    }

    // 获取BGR 3D向量映射
    pub fn get_bgr_vector3c_map(&self) -> Vector3cMap {
        Vector3cMap::new(self.b, self.g, self.r)
    }

    // 获取BGRA 4D向量映射
    pub fn get_bgra_vector4c_map(&self) -> Vector4cMap {
        Vector4cMap::new(self.b, self.g, self.r, self.a)
    }
}

// 4. 定义INTENSITY结构体
pub struct PclAddIntensity {
    pub intensity: f32,
}

// 5. 定义INTENSITY_8U结构体
pub struct PclAddIntensity8u {
    pub intensity: u8,
}

// 6. 定义INTENSITY_32U结构体
pub struct PclAddIntensity32u {
    pub intensity: u32,
}

/* end 线性代数适配nalgebra库 */

#[cfg(test)]
mod tests1 {
    use super::*;

    // 测试PclAddPoint4D结构体的向量映射功能
    #[test]
    fn test_pcl_add_point4d() {
        let point = PclAddPoint4D {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            data: [1.0, 2.0, 3.0, 4.0],
        };

        // 测试2D向量映射
        let vector2f = point.get_vector2f_map();
        assert_eq!(vector2f.x, 1.0);
        assert_eq!(vector2f.y, 2.0);

        // 测试3D向量映射
        let vector3f = point.get_vector3f_map();
        assert_eq!(vector3f.x, 1.0);
        assert_eq!(vector3f.y, 2.0);
        assert_eq!(vector3f.z, 3.0);

        // 测试4D向量映射
        let vector4f = point.get_vector4f_map();
        assert_eq!(vector4f.x, 1.0);
        assert_eq!(vector4f.y, 2.0);
        assert_eq!(vector4f.z, 3.0);
        assert_eq!(vector4f.w, 4.0);
    }

    // 测试PclAddNormal4D结构体的法线向量映射功能
    #[test]
    fn test_pcl_add_normal4d() {
        let normal = PclAddNormal4D {
            normal_x: 1.0,
            normal_y: 2.0,
            normal_z: 3.0,
            data_n: [1.0, 2.0, 3.0, 4.0],
            normal: [1.0, 2.0, 3.0],
        };

        // 测试3D法线向量映射
        let normal_vector3f = normal.get_normal_vector3f_map();
        assert_eq!(normal_vector3f.x, 1.0);
        assert_eq!(normal_vector3f.y, 2.0);
        assert_eq!(normal_vector3f.z, 3.0);

        // 测试4D法线向量映射
        let normal_vector4f = normal.get_normal_vector4f_map();
        assert_eq!(normal_vector4f.x, 1.0);
        assert_eq!(normal_vector4f.y, 2.0);
        assert_eq!(normal_vector4f.z, 3.0);
        assert_eq!(normal_vector4f.w, 4.0);
    }

    // 测试PclAddRGB结构体的颜色向量映射功能
    #[test]
    fn test_pcl_add_rgb() {
        let rgb = PclAddRGB {
            r: 255,
            g: 128,
            b: 64,
            a: 32,
            rgba: 0xFF804020,
        };

        // 测试RGB 3D向量
        let rgb_vector3i = rgb.get_rgb_vector3i();
        assert_eq!(rgb_vector3i.x, 255);
        assert_eq!(rgb_vector3i.y, 128);
        assert_eq!(rgb_vector3i.z, 64);

        // 测试RGBA 4D向量
        let rgba_vector4i = rgb.get_rgba_vector4i();
        assert_eq!(rgba_vector4i.x, 255);
        assert_eq!(rgba_vector4i.y, 128);
        assert_eq!(rgba_vector4i.z, 64);
        assert_eq!(rgba_vector4i.w, 32);

        // 测试BGR 3D向量映射
        let bgr_vector3c = rgb.get_bgr_vector3c_map();
        assert_eq!(bgr_vector3c.x, 64);
        assert_eq!(bgr_vector3c.y, 128);
        assert_eq!(bgr_vector3c.z, 255);

        // 测试BGRA 4D向量映射
        let bgra_vector4c = rgb.get_bgra_vector4c_map();
        assert_eq!(bgra_vector4c.x, 64);
        assert_eq!(bgra_vector4c.y, 128);
        assert_eq!(bgra_vector4c.z, 255);
        assert_eq!(bgra_vector4c.w, 32);
    }
}
