#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]

//! 点结构体

// 原子数据(空数据)
use std::marker::PhantomData;

// 异常处理
use anyhow::Result;
// 浮点数精度处理
use approx;
// 二进制编码
use base64;
// 命令行参数解析
use cpal;
// 性能评测
use criterion;
// kdtree数据结构
use kd_tree;
// 线性代数库
use nalgebra::DMatrix;
// 复数支持
use num_complex;
// 数学特性
use num_traits;
// 宏编程/元编程
use paste;
use proc_macro2;
use quote::quote;
use syn::{Ident, Type};
// 随机数
use rand::{self, Rng};
// 随机数分布
use rand_distr;
// fft快速傅立叶变换
use realfft;
// 机器学习,数据分析
use rstats;
// 序列化
use serde::{Deserialize, Serialize};
// 多线程
use tokio;
// PCD文件处理
use bye_pcd_rs::{PcdDeserialize, PcdSerialize};

// 自定义序列化
// use crate::common::point_types::my_serde::{MyDeserialize, MySerialize};
/* start 点云结构体 */

// 1. 成员: float x, y, z
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointXYZ {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

// 实现Display trait，用于格式化输出
impl std::fmt::Display for PointXYZ {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{},{})", self.x, self.y, self.z)
    }
}

// 2. 成员: rgba
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct RGB {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl std::fmt::Display for RGB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{},{},{})", self.r, self.g, self.b, self.a)
    }
}

// 3. 成员: intensity (float)
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct Intensity {
    pub intensity: f32,
}

impl std::fmt::Display for Intensity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({})", self.intensity)
    }
}

// 4. 成员: intensity (u8)
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct Intensity8u {
    pub intensity: u8,
}

impl std::fmt::Display for Intensity8u {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({})", self.intensity)
    }
}

// 5. 成员: intensity (u32)
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct Intensity32u {
    pub intensity: u32,
}

impl std::fmt::Display for Intensity32u {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({})", self.intensity)
    }
}

// 6. 成员: float x, y, z, intensity
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointXYZI {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
}

impl std::fmt::Display for PointXYZI {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{},{} - {})", self.x, self.y, self.z, self.intensity)
    }
}

// 7. 成员: float x, y, z, u32 label
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointXYZL {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub label: u32,
}

impl std::fmt::Display for PointXYZL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{},{} - {})", self.x, self.y, self.z, self.label)
    }
}

// 8. 成员: u32 label
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct Label {
    pub label: u32,
}

impl std::fmt::Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({})", self.label)
    }
}

// 9. 成员: float x, y, z; u32 rgba
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointXYZRGBA {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub rgba: u32,
}

impl std::fmt::Display for PointXYZRGBA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {},{},{},{})",
            self.x,
            self.y,
            self.z,
            (self.rgba >> 24) & 0xFF,
            (self.rgba >> 16) & 0xFF,
            (self.rgba >> 8) & 0xFF,
            self.rgba & 0xFF
        )
    }
}

// 10. 成员: float x, y, z, rgb
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointXYZRGB {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub rgb: u32,
}

impl std::fmt::Display for PointXYZRGB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {},{},{})",
            self.x,
            self.y,
            self.z,
            (self.rgb >> 16) & 0xFF,
            (self.rgb >> 8) & 0xFF,
            self.rgb & 0xFF
        )
    }
}

// 11. 成员: float x, y, z, rgb, u32 label
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointXYZRGBL {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub rgb: u32,
    pub label: u32,
}

impl std::fmt::Display for PointXYZRGBL {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {},{},{} - {})",
            self.x,
            self.y,
            self.z,
            (self.rgb >> 16) & 0xFF,
            (self.rgb >> 8) & 0xFF,
            self.rgb & 0xFF,
            self.label
        )
    }
}

// 12. 成员: float x, y, z, L, a, b
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointXYZLAB {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

impl std::fmt::Display for PointXYZLAB {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {}, {}, {})",
            self.x, self.y, self.z, self.l, self.a, self.b
        )
    }
}

// 13. 成员: float x, y, z, h, s, v
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointXYZHSV {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub h: f32,
    pub s: f32,
    pub v: f32,
}

impl std::fmt::Display for PointXYZHSV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {}, {}, {})",
            self.x, self.y, self.z, self.h, self.s, self.v
        )
    }
}

// 14. 成员: float x, y
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointXY {
    pub x: f32,
    pub y: f32,
}

impl std::fmt::Display for PointXY {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.x, self.y)
    }
}

// 15. 成员: float u, v
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointUV {
    pub u: f32,
    pub v: f32,
}

impl std::fmt::Display for PointUV {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.u, self.v)
    }
}

// 16. 成员: float x, y, z, strength
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct InterestPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub strength: f32,
}

impl std::fmt::Display for InterestPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{},{} - {})", self.x, self.y, self.z, self.strength)
    }
}

// 17. 成员: float normal[3], curvature
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct Normal {
    pub normal: [f32; 3],
    pub curvature: f32,
}

impl std::fmt::Display for Normal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {})",
            self.normal[0], self.normal[1], self.normal[2], self.curvature
        )
    }
}

// 18. 成员: float normal[3]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, PcdDeserialize, PcdSerialize)]
pub struct Axis {
    pub normal: [f32; 3],
}

impl std::fmt::Display for Axis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{})",
            self.normal[0], self.normal[1], self.normal[2]
        )
    }
}

// 19. 成员: float x, y, z; float normal[3], curvature
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointNormal {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub normal: [f32; 3],
    pub curvature: f32,
}

impl std::fmt::Display for PointNormal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {},{},{} - {})",
            self.x, self.y, self.z, self.normal[0], self.normal[1], self.normal[2], self.curvature
        )
    }
}

// 20. 成员: float x, y, z, rgb, normal[3], curvature
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointXYZRGBNormal {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub rgb: u32,
    pub normal: [f32; 3],
    pub curvature: f32,
}

impl std::fmt::Display for PointXYZRGBNormal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {},{},{} - {},{},{} - {})",
            self.x,
            self.y,
            self.z,
            (self.rgb >> 16) & 0xFF,
            (self.rgb >> 8) & 0xFF,
            self.rgb & 0xFF,
            self.normal[0],
            self.normal[1],
            self.normal[2],
            self.curvature
        )
    }
}

// 21. 成员: float x, y, z, intensity, normal[3], curvature
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointXYZINormal {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub normal: [f32; 3],
    pub curvature: f32,
}

impl std::fmt::Display for PointXYZINormal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {} - {},{},{} - {})",
            self.x,
            self.y,
            self.z,
            self.intensity,
            self.normal[0],
            self.normal[1],
            self.normal[2],
            self.curvature
        )
    }
}

// 22. 成员: float x, y, z, label, normal[3], curvature
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointXYZLNormal {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub label: u32,
    pub normal: [f32; 3],
    pub curvature: f32,
}

impl std::fmt::Display for PointXYZLNormal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {} - {},{},{} - {})",
            self.x,
            self.y,
            self.z,
            self.label,
            self.normal[0],
            self.normal[1],
            self.normal[2],
            self.curvature
        )
    }
}

// 23. 成员: float x, y, z (union with float point[4]), range
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointWithRange {
    pub point: [f32; 4],
    pub range: f32,
}

impl std::fmt::Display for PointWithRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {})",
            self.point[0], self.point[1], self.point[2], self.range
        )
    }
}

// 24. 成员: float x, y, z, vp_x, vp_y, vp_z
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointWithViewpoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub vp_x: f32,
    pub vp_y: f32,
    pub vp_z: f32,
}

impl std::fmt::Display for PointWithViewpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {},{},{})",
            self.x, self.y, self.z, self.vp_x, self.vp_y, self.vp_z
        )
    }
}

// 25. 成员: float j1, j2, j3
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct MomentInvariants {
    pub j1: f32,
    pub j2: f32,
    pub j3: f32,
}

impl std::fmt::Display for MomentInvariants {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{},{})", self.j1, self.j2, self.j3)
    }
}

// 26. 成员: float r_min, r_max
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PrincipalRadiiRSD {
    pub r_min: f32,
    pub r_max: f32,
}

impl std::fmt::Display for PrincipalRadiiRSD {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.r_min, self.r_max)
    }
}

// 27. 成员: u8 boundary_point
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct Boundary {
    pub boundary_point: u8,
}

impl std::fmt::Display for Boundary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({})", self.boundary_point)
    }
}

// 28. 成员: float principal_curvature[3], pc1, pc2
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PrincipalCurvatures {
    pub principal_curvature: [f32; 3],
    pub pc1: f32,
    pub pc2: f32,
}

impl std::fmt::Display for PrincipalCurvatures {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {},{})",
            self.principal_curvature[0],
            self.principal_curvature[1],
            self.principal_curvature[2],
            self.pc1,
            self.pc2
        )
    }
}

// 29. 成员: float descriptor[352], rf[9]
// Serialize无法支持352数组序列化
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct SHOT352 {
    pub descriptor: [f32; 352],
    pub rf: [f32; 9],
}

impl std::fmt::Display for SHOT352 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for i in 0..9 {
            write!(f, "{}", self.rf[i])?;
            if i < 8 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")?;
        for i in 0..352 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.descriptor[i])?;
            if i < 351 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 30. 成员: float descriptor[1344], rf[9]
// Serialize无法支持1344数组序列化
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct SHOT1344 {
    pub descriptor: [f32; 1344],
    pub rf: [f32; 9],
}

impl std::fmt::Display for SHOT1344 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for i in 0..9 {
            write!(f, "{}", self.rf[i])?;
            if i < 8 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")?;
        for i in 0..1344 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.descriptor[i])?;
            if i < 1343 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 31. 成员: Axis x_axis, y_axis, z_axis
// Serialize无法支持大数组序列化
#[derive(Debug, Clone)]
pub struct ReferenceFrame {
    pub x_axis: Axis,
    pub y_axis: Axis,
    pub z_axis: Axis,
}

impl std::fmt::Display for ReferenceFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({} {} {},{} {} {},{} {} {})",
            self.x_axis.normal[0],
            self.x_axis.normal[1],
            self.x_axis.normal[2],
            self.y_axis.normal[0],
            self.y_axis.normal[1],
            self.y_axis.normal[2],
            self.z_axis.normal[0],
            self.z_axis.normal[1],
            self.z_axis.normal[2]
        )
    }
}

// 32. 成员: float descriptor[1980], rf[9]
// Serialize无法支持大数组序列化
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct ShapeContext1980 {
    pub descriptor: [f32; 1980],
    pub rf: [f32; 9],
}

impl std::fmt::Display for ShapeContext1980 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for i in 0..9 {
            write!(f, "{}", self.rf[i])?;
            if i < 8 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")?;
        for i in 0..1980 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.descriptor[i])?;
            if i < 1979 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 33. 成员: float descriptor[1960], rf[9]
// Serialize无法支持大数组序列化
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct UniqueShapeContext1960 {
    pub descriptor: [f32; 1960],
    pub rf: [f32; 9],
}

impl std::fmt::Display for UniqueShapeContext1960 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "(")?;
        for i in 0..9 {
            write!(f, "{}", self.rf[i])?;
            if i < 8 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")?;
        for i in 0..1960 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.descriptor[i])?;
            if i < 1959 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 34. 成员: float pfh[125]
// Serialize无法支持大数组序列化
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PFHSignature125 {
    pub pfh: [f32; 125],
}

impl std::fmt::Display for PFHSignature125 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "(")?;
        for i in 0..125 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.pfh[i])?;
            if i < 124 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 35. 成员: float pfhrgb[250]
// Serialize无法支持大数组序列化
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct PFHRGBSignature250 {
    pub pfhrgb: [f32; 250],
}

impl std::fmt::Display for PFHRGBSignature250 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "(")?;
        for i in 0..250 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.pfhrgb[i])?;
            if i < 249 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 36. 成员: float f1, f2, f3, f4, alpha_m
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PPFSignature {
    pub f1: f32,
    pub f2: f32,
    pub f3: f32,
    pub f4: f32,
    pub alpha_m: f32,
}

impl std::fmt::Display for PPFSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}, {}, {}, {}, {})",
            self.f1, self.f2, self.f3, self.f4, self.alpha_m
        )
    }
}

// 37. 成员: float f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, alpha_m
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct CPPFSignature {
    pub f1: f32,
    pub f2: f32,
    pub f3: f32,
    pub f4: f32,
    pub f5: f32,
    pub f6: f32,
    pub f7: f32,
    pub f8: f32,
    pub f9: f32,
    pub f10: f32,
    pub alpha_m: f32,
}

impl std::fmt::Display for CPPFSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})",
            self.f1,
            self.f2,
            self.f3,
            self.f4,
            self.f5,
            self.f6,
            self.f7,
            self.f8,
            self.f9,
            self.f10,
            self.alpha_m
        )
    }
}

// 38. 成员: float f1, f2, f3, f4, r_ratio, g_ratio, b_ratio, alpha_m
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PPFRGBSignature {
    pub f1: f32,
    pub f2: f32,
    pub f3: f32,
    pub f4: f32,
    pub r_ratio: f32,
    pub g_ratio: f32,
    pub b_ratio: f32,
    pub alpha_m: f32,
}

impl std::fmt::Display for PPFRGBSignature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}, {}, {}, {}, {}, {}, {}, {})",
            self.f1,
            self.f2,
            self.f3,
            self.f4,
            self.r_ratio,
            self.g_ratio,
            self.b_ratio,
            self.alpha_m
        )
    }
}

// 39. 成员: float values[12]
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
#[serde(transparent)]
pub struct NormalBasedSignature12 {
    pub values: [f32; 12],
}

impl std::fmt::Display for NormalBasedSignature12 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "(")?;
        for i in 0..12 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.values[i])?;
            if i < 11 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 40. 成员: float fpfh[33]
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct FPFHSignature33 {
    pub fpfh: [f32; 33],
}

impl std::fmt::Display for FPFHSignature33 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "(")?;
        for i in 0..33 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.fpfh[i])?;
            if i < 32 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 41. 成员: float vfh[308]
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct VFHSignature308 {
    pub vfh: [f32; 308],
}

impl std::fmt::Display for VFHSignature308 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "(")?;
        for i in 0..308 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.vfh[i])?;
            if i < 307 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 42. 成员: float grsd[21]
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct GRSDSignature21 {
    pub grsd: [f32; 21],
}

impl std::fmt::Display for GRSDSignature21 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "(")?;
        for i in 0..21 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.grsd[i])?;
            if i < 20 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 43. 成员: float esf[640]
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct ESFSignature640 {
    pub esf: [f32; 640],
}

impl std::fmt::Display for ESFSignature640 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "(")?;
        for i in 0..640 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.esf[i])?;
            if i < 639 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 44. 成员: float gasd[512]
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct GASDSignature512 {
    pub gasd: [f32; 512],
}

impl std::fmt::Display for GASDSignature512 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "(")?;
        for i in 0..512 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.gasd[i])?;
            if i < 511 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 45. 成员: float gasd[984]
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct GASDSignature984 {
    pub gasd: [f32; 984],
}

impl std::fmt::Display for GASDSignature984 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "(")?;
        for i in 0..984 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.gasd[i])?;
            if i < 983 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 46. 成员: float gasd[7992]
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct GASDSignature7992 {
    pub gasd: [f32; 7992],
}

impl std::fmt::Display for GASDSignature7992 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "(")?;
        for i in 0..7992 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.gasd[i])?;
            if i < 7991 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 47. 成员: float histogram[16]
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct GFPFHSignature16 {
    pub gfpfh: [f32; 16],
}

impl std::fmt::Display for GFPFHSignature16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "(")?;
        for i in 0..16 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.gfpfh[i])?;
            if i < 15 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 48. 成员: float scale; float orientation; u8 descriptor[64]
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct BRISKSignature512 {
    pub scale: f32,
    pub orientation: f32,
    pub descriptor: [u8; 64],
}

impl std::fmt::Display for BRISKSignature512 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} ", self.scale, self.orientation)?;
        // write!(f, "(")?;
        for i in 0..64 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.descriptor[i])?;
            if i < 63 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 49. 成员: float x, y, z, roll, pitch, yaw; float descriptor[36]
#[derive(Debug, Clone, PcdDeserialize, PcdSerialize)]
pub struct Narf36 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub roll: f32,
    pub pitch: f32,
    pub yaw: f32,
    pub descriptor: [f32; 36],
}

impl std::fmt::Display for Narf36 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{},{},{} - {}deg,{}deg,{}deg - ",
            self.x,
            self.y,
            self.z,
            self.roll * 360.0 / std::f32::consts::PI,
            self.pitch * 360.0 / std::f32::consts::PI,
            self.yaw * 360.0 / std::f32::consts::PI
        )?;
        // write!(f, "(")?;
        for i in 0..36 {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.descriptor[i])?;
            if i < 35 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 使用固定大小的位集合来表示边界特征
pub type BorderTraits = [bool; 32];

// 50. 成员: int x, y; BorderTraits traits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderDescription {
    pub x: i32,
    pub y: i32,
    pub traits: BorderTraits,
}

impl std::fmt::Display for BorderDescription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.x, self.y)
    }
}

// 51. 成员: float gradient[3]
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct IntensityGradient {
    pub gradient: [f32; 3],
}

impl std::fmt::Display for IntensityGradient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{})",
            self.gradient[0], self.gradient[1], self.gradient[2],
        )
    }
}

// 52. 成员: float histogram[N]
#[derive(Debug, Clone)]
pub struct Histogram<const N: usize> {
    pub histogram: [f32; N],
}

impl<const N: usize> std::fmt::Display for Histogram<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // write!(f, "(")?;
        for i in 0..N {
            if i == 0 {
                write!(f, "(")?;
            }
            write!(f, "{}", self.histogram[i])?;
            if i < N - 1 {
                write!(f, ", ")?;
            }
        }
        write!(f, ")")
    }
}

// 53. 成员: float x, y, z, scale, angle, response, octave
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointWithScale {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub scale: f32,
    pub angle: f32,
    pub response: f32,
    pub octave: f32,
}

impl std::fmt::Display for PointWithScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {},{},{},{})",
            self.x, self.y, self.z, self.scale, self.angle, self.response, self.octave
        )
    }
}

// 54. 成员: float x, y, z, normal[3], rgba, radius, confidence, curvature
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointSurfel {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub normal: [f32; 3],
    pub rgba: u32,
    pub radius: f32,
    pub confidence: f32,
    pub curvature: f32,
}

impl std::fmt::Display for PointSurfel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {},{},{} - {},{},{},{} - {} - {} - {})",
            self.x,
            self.y,
            self.z,
            self.normal[0],
            self.normal[1],
            self.normal[2],
            (self.rgba >> 24) & 0xFF,
            (self.rgba >> 16) & 0xFF,
            (self.rgba >> 8) & 0xFF,
            self.rgba & 0xFF,
            self.radius,
            self.confidence,
            self.curvature
        )
    }
}

// 55. 成员: float x, y, z, intensity, intensity_variance, height_variance
#[derive(Debug, Clone, Serialize, Deserialize, PcdDeserialize, PcdSerialize)]
pub struct PointDEM {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub intensity_variance: f32,
    pub height_variance: f32,
}

impl std::fmt::Display for PointDEM {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({},{},{} - {} - {} - {})",
            self.x, self.y, self.z, self.intensity, self.intensity_variance, self.height_variance
        )
    }
}
/* end 点云结构体 */

/* start 点云枚举 */

// 1. 定义所有的具有XYZ和标签的点云
pub enum PCL_XYZL_POINT_TYPES {
    PointXYZL,
    PointXYZRGBL,
    PointXYZLNormal,
}

// 2. 定义所有包含normal[3]数据的点云
pub enum PCL_NORMAL_POINT_TYPES {
    Normal,
    PointNormal,
    PointXYZRGBNormal,
    PointXYZINormal,
    PointXYZLNormal,
    PointSurfel,
}

// 3. 定义所有表示特征的点云
pub enum PCL_FEATURE_POINT_TYPES {
    PFHSignature125,
    PFHRGBSignature250,
    PPFSignature,
    CPPFSignature,
    PPFRGBSignature,
    NormalBasedSignature12,
    FPFHSignature33,
    VFHSignature308,
    GASDSignature512,
    GASDSignature984,
    GASDSignature7992,
    GRSDSignature21,
    ESFSignature640,
    BRISKSignature512,
    Narf36,
}

// 4. 定义所有具有descriptorSize()成员函数的点云
pub enum PCL_DESCRIPTOR_FEATURE_POINT_TYPES {
    PFHSignature125,
    PFHRGBSignature250,
    FPFHSignature33,
    VFHSignature308,
    GASDSignature512,
    GASDSignature984,
    GASDSignature7992,
    GRSDSignature21,
    ESFSignature640,
    BRISKSignature512,
    Narf36,
}

/* end 点云枚举 */

#[cfg(test)]
mod tests1 {
    use super::*;

    // 测试 PointXYZ 结构体
    #[test]
    fn test_point_xyz() {
        let point = PointXYZ {
            x: 1.0,
            y: 2.0,
            z: 3.0,
        };
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(format!("{}", point), "(1,2,3)");
    }

    // 测试 RGB 结构体
    #[test]
    fn test_rgb() {
        let color = RGB {
            r: 255,
            g: 0,
            b: 128,
            a: 255,
        };
        assert_eq!(color.r, 255);
        assert_eq!(color.g, 0);
        assert_eq!(color.b, 128);
        assert_eq!(color.a, 255);
        assert_eq!(format!("{}", color), "(255,0,128,255)");
    }

    // 测试 Intensity 结构体
    #[test]
    fn test_intensity() {
        let intensity = Intensity { intensity: 0.5 };
        assert_eq!(intensity.intensity, 0.5);
        assert_eq!(format!("{}", intensity), "(0.5)");
    }

    // 测试 Intensity8u 结构体
    #[test]
    fn test_intensity8u() {
        let intensity = Intensity8u { intensity: 128 };
        assert_eq!(intensity.intensity, 128);
        assert_eq!(format!("{}", intensity), "(128)");
    }

    // 测试 Intensity32u 结构体
    #[test]
    fn test_intensity32u() {
        let intensity = Intensity32u {
            intensity: 4294967295,
        };
        assert_eq!(intensity.intensity, 4294967295);
        assert_eq!(format!("{}", intensity), "(4294967295)");
    }

    // 测试 PointXYZI 结构体
    #[test]
    fn test_point_xyzi() {
        let point = PointXYZI {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            intensity: 0.5,
        };
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.intensity, 0.5);
        assert_eq!(format!("{}", point), "(1,2,3 - 0.5)");
    }

    // 测试 PointXYZL 结构体
    #[test]
    fn test_point_xyzl() {
        let point = PointXYZL {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            label: 42,
        };
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.label, 42);
        assert_eq!(format!("{}", point), "(1,2,3 - 42)");
    }

    // 测试 Label 结构体
    #[test]
    fn test_label() {
        let label = Label { label: 42 };
        assert_eq!(label.label, 42);
        assert_eq!(format!("{}", label), "(42)");
    }

    // 测试 PointXYZRGBA 结构体
    #[test]
    fn test_point_xyzrgba() {
        let point = PointXYZRGBA {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            rgba: 0xFF00FF80,
        };
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.rgba, 0xFF00FF80);
        assert_eq!(format!("{}", point), "(1,2,3 - 255,0,255,128)");
    }

    // 测试 PointXYZRGB 结构体
    #[test]
    fn test_point_xyzrgb() {
        let point = PointXYZRGB {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            rgb: 0xFF00FF,
        };
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.rgb, 0xFF00FF);
        assert_eq!(format!("{}", point), "(1,2,3 - 255,0,255)");
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;

    // 测试 PointXYZRGBL 结构体
    #[test]
    fn test_point_xyzrgb_l() {
        let point = PointXYZRGBL {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            rgb: 0xFFAABB,
            label: 42,
        };
        // 测试 Display trait 实现
        assert_eq!(point.to_string(), "(1,2,3 - 255,170,187 - 42)");
    }

    // 测试 PointXYZLAB 结构体
    #[test]
    fn test_point_xyz_lab() {
        let point = PointXYZLAB {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            l: 0.5,
            a: 0.6,
            b: 0.7,
        };
        // 测试 Display trait 实现
        assert_eq!(point.to_string(), "(1,2,3 - 0.5, 0.6, 0.7)");
    }

    // 测试 PointXYZHSV 结构体
    #[test]
    fn test_point_xyz_hsv() {
        let point = PointXYZHSV {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            h: 0.1,
            s: 0.2,
            v: 0.3,
        };
        // 测试 Display trait 实现
        assert_eq!(point.to_string(), "(1,2,3 - 0.1, 0.2, 0.3)");
    }

    // 测试 PointXY 结构体
    #[test]
    fn test_point_xy() {
        let point = PointXY { x: 1.0, y: 2.0 };
        // 测试 Display trait 实现
        assert_eq!(point.to_string(), "(1,2)");
    }

    // 测试 PointUV 结构体
    #[test]
    fn test_point_uv() {
        let point = PointUV { u: 0.1, v: 0.2 };
        // 测试 Display trait 实现
        assert_eq!(point.to_string(), "(0.1,0.2)");
    }

    // 测试 InterestPoint 结构体
    #[test]
    fn test_interest_point() {
        let point = InterestPoint {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            strength: 0.5,
        };
        // 测试 Display trait 实现
        assert_eq!(point.to_string(), "(1,2,3 - 0.5)");
    }

    // 测试 Normal 结构体
    #[test]
    fn test_normal() {
        let normal = Normal {
            normal: [0.1, 0.2, 0.3],
            curvature: 0.5,
        };
        // 测试 Display trait 实现
        assert_eq!(normal.to_string(), "(0.1,0.2,0.3 - 0.5)");
    }

    // 测试 Axis 结构体
    #[test]
    fn test_axis() {
        let axis = Axis {
            normal: [0.1, 0.2, 0.3],
        };
        // 测试 Display trait 实现
        assert_eq!(axis.to_string(), "(0.1,0.2,0.3)");
    }

    // 测试 PointNormal 结构体
    #[test]
    fn test_point_normal() {
        let point = PointNormal {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            normal: [0.1, 0.2, 0.3],
            curvature: 0.5,
        };
        // 测试 Display trait 实现
        assert_eq!(point.to_string(), "(1,2,3 - 0.1,0.2,0.3 - 0.5)");
    }

    // 测试 PointXYZRGBNormal 结构体
    #[test]
    fn test_point_xyz_rgb_normal() {
        let point = PointXYZRGBNormal {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            rgb: 0xFFAABB,
            normal: [0.1, 0.2, 0.3],
            curvature: 0.5,
        };
        // 测试 Display trait 实现
        assert_eq!(
            point.to_string(),
            "(1,2,3 - 255,170,187 - 0.1,0.2,0.3 - 0.5)"
        );
    }
}

#[cfg(test)]
mod tests3 {
    use super::*;

    // 测试 PointXYZINormal 结构体的显示功能
    #[test]
    fn test_point_xyz_i_normal_display() {
        let point = PointXYZINormal {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            intensity: 0.5,
            normal: [0.1, 0.2, 0.3],
            curvature: 0.01,
        };
        assert_eq!(format!("{}", point), "(1,2,3 - 0.5 - 0.1,0.2,0.3 - 0.01)");
    }

    // 测试 PointXYZLNormal 结构体的显示功能
    #[test]
    fn test_point_xyz_l_normal_display() {
        let point = PointXYZLNormal {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            label: 42,
            normal: [0.1, 0.2, 0.3],
            curvature: 0.01,
        };
        assert_eq!(format!("{}", point), "(1,2,3 - 42 - 0.1,0.2,0.3 - 0.01)");
    }

    // 测试 PointWithRange 结构体的显示功能
    #[test]
    fn test_point_with_range_display() {
        let point = PointWithRange {
            point: [1.0, 2.0, 3.0, 0.0],
            range: 10.0,
        };
        assert_eq!(format!("{}", point), "(1,2,3 - 10)");
    }

    // 测试 PointWithViewpoint 结构体的显示功能
    #[test]
    fn test_point_with_viewpoint_display() {
        let point = PointWithViewpoint {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            vp_x: 4.0,
            vp_y: 5.0,
            vp_z: 6.0,
        };
        assert_eq!(format!("{}", point), "(1,2,3 - 4,5,6)");
    }

    // 测试 MomentInvariants 结构体的显示功能
    #[test]
    fn test_moment_invariants_display() {
        let moment = MomentInvariants {
            j1: 1.0,
            j2: 2.0,
            j3: 3.0,
        };
        assert_eq!(format!("{}", moment), "(1,2,3)");
    }

    // 测试 PrincipalRadiiRSD 结构体的显示功能
    #[test]
    fn test_principal_radii_rsd_display() {
        let radii = PrincipalRadiiRSD {
            r_min: 1.0,
            r_max: 2.0,
        };
        assert_eq!(format!("{}", radii), "(1,2)");
    }

    // 测试 Boundary 结构体的显示功能
    #[test]
    fn test_boundary_display() {
        let boundary = Boundary { boundary_point: 1 };
        assert_eq!(format!("{}", boundary), "(1)");
    }

    // 测试 PrincipalCurvatures 结构体的显示功能
    #[test]
    fn test_principal_curvatures_display() {
        let curvatures = PrincipalCurvatures {
            principal_curvature: [0.1, 0.2, 0.3],
            pc1: 1.0,
            pc2: 2.0,
        };
        assert_eq!(format!("{}", curvatures), "(0.1,0.2,0.3 - 1,2)");
    }

    // 测试 SHOT352 结构体的显示功能
    #[test]
    fn test_shot352_display() {
        let shot = SHOT352 {
            descriptor: [0.0; 352],
            rf: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let expected_rf = "(1, 2, 3, 4, 5, 6, 7, 8, 9)";
        let expected_descriptor = "(".to_string() + &"0, ".repeat(351) + "0)";
        assert!(format!("{}", shot).starts_with(expected_rf));
        assert!(format!("{}", shot).ends_with(&expected_descriptor));
    }

    // 测试 SHOT1344 结构体的显示功能
    #[test]
    fn test_shot1344_display() {
        let shot = SHOT1344 {
            descriptor: [0.0; 1344],
            rf: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let expected_rf = "(1, 2, 3, 4, 5, 6, 7, 8, 9)";
        let expected_descriptor = "(".to_string() + &"0, ".repeat(1343) + "0)";
        assert!(format!("{}", shot).starts_with(expected_rf));
        assert!(format!("{}", shot).ends_with(&expected_descriptor));
    }
}

#[cfg(test)]
mod tests4 {
    use super::*;

    // 测试 ReferenceFrame 结构体的显示功能
    #[test]
    fn test_reference_frame_display() {
        let frame = ReferenceFrame {
            x_axis: Axis {
                normal: [1.0, 0.0, 0.0],
            },
            y_axis: Axis {
                normal: [0.0, 1.0, 0.0],
            },
            z_axis: Axis {
                normal: [0.0, 0.0, 1.0],
            },
        };
        assert_eq!(format!("{}", frame), "(1 0 0,0 1 0,0 0 1)");
    }

    // 测试 ShapeContext1980 结构体的显示功能
    #[test]
    fn test_shape_context_1980_display() {
        let context = ShapeContext1980 {
            descriptor: [0.0; 1980],
            rf: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let expected_rf = "(1, 2, 3, 4, 5, 6, 7, 8, 9)";
        let expected_descriptor = "(".to_string() + &"0, ".repeat(1979) + "0)";
        assert!(format!("{}", context).starts_with(expected_rf));
        assert!(format!("{}", context).ends_with(&expected_descriptor));
    }

    // 测试 UniqueShapeContext1960 结构体的显示功能
    #[test]
    fn test_unique_shape_context_1960_display() {
        let context = UniqueShapeContext1960 {
            descriptor: [0.0; 1960],
            rf: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        };
        let expected_rf = "(1, 2, 3, 4, 5, 6, 7, 8, 9)";
        let expected_descriptor = "(".to_string() + &"0, ".repeat(1959) + "0)";
        assert!(format!("{}", context).starts_with(expected_rf));
        assert!(format!("{}", context).ends_with(&expected_descriptor));
    }

    // 测试 PFHSignature125 结构体的显示功能
    #[test]
    fn test_pfh_signature_125_display() {
        let signature = PFHSignature125 { pfh: [0.0; 125] };
        let expected = "(".to_string() + &"0, ".repeat(124) + "0)";
        assert_eq!(format!("{}", signature), expected);
    }

    // 测试 PFHRGBSignature250 结构体的显示功能
    #[test]
    fn test_pfhrgb_signature_250_display() {
        let signature = PFHRGBSignature250 { pfhrgb: [0.0; 250] };
        let expected = "(".to_string() + &"0, ".repeat(249) + "0)";
        assert_eq!(format!("{}", signature), expected);
    }

    // 测试 PPFSignature 结构体的显示功能
    #[test]
    fn test_ppf_signature_display() {
        let signature = PPFSignature {
            f1: 1.0,
            f2: 2.0,
            f3: 3.0,
            f4: 4.0,
            alpha_m: 0.5,
        };
        assert_eq!(format!("{}", signature), "(1, 2, 3, 4, 0.5)");
    }

    // 测试 CPPFSignature 结构体的显示功能
    #[test]
    fn test_cppf_signature_display() {
        let signature = CPPFSignature {
            f1: 1.0,
            f2: 2.0,
            f3: 3.0,
            f4: 4.0,
            f5: 5.0,
            f6: 6.0,
            f7: 7.0,
            f8: 8.0,
            f9: 9.0,
            f10: 10.0,
            alpha_m: 0.5,
        };
        assert_eq!(
            format!("{}", signature),
            "(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0.5)"
        );
    }

    // 测试 PPFRGBSignature 结构体的显示功能
    #[test]
    fn test_ppfrgb_signature_display() {
        let signature = PPFRGBSignature {
            f1: 1.0,
            f2: 2.0,
            f3: 3.0,
            f4: 4.0,
            r_ratio: 0.1,
            g_ratio: 0.2,
            b_ratio: 0.3,
            alpha_m: 0.5,
        };
        assert_eq!(format!("{}", signature), "(1, 2, 3, 4, 0.1, 0.2, 0.3, 0.5)");
    }

    // 测试 NormalBasedSignature12 结构体的显示功能
    #[test]
    fn test_normal_based_signature_12_display() {
        let signature = NormalBasedSignature12 { values: [0.0; 12] };
        let expected = "(".to_string() + &"0, ".repeat(11) + "0)";
        assert_eq!(format!("{}", signature), expected);
    }

    // 测试 FPFHSignature33 结构体的显示功能
    #[test]
    fn test_fpfh_signature_33_display() {
        let signature = FPFHSignature33 { fpfh: [0.0; 33] };
        let expected = "(".to_string() + &"0, ".repeat(32) + "0)";
        assert_eq!(format!("{}", signature), expected);
    }
}

#[cfg(test)]
mod tests5 {
    use super::*;

    #[test]
    fn test_vfh_signature_308() {
        // 测试 VFHSignature308 的 Display 实现
        let vfh = VFHSignature308 { vfh: [0.0; 308] };
        let output = format!("{}", vfh);
        assert!(output.starts_with("(") && output.ends_with(")"));
        assert_eq!(output.matches(", ").count(), 307); // 确保有 307 个逗号
    }

    #[test]
    fn test_grsd_signature_21() {
        // 测试 GRSDSignature21 的 Display 实现
        let grsd = GRSDSignature21 { grsd: [1.0; 21] };
        let output = format!("{}", grsd);
        assert!(output.starts_with("(") && output.ends_with(")"));
        assert_eq!(output.matches(", ").count(), 20); // 确保有 20 个逗号
    }

    #[test]
    fn test_esf_signature_640() {
        // 测试 ESFSignature640 的 Display 实现
        let esf = ESFSignature640 { esf: [2.0; 640] };
        let output = format!("{}", esf);
        assert!(output.starts_with("(") && output.ends_with(")"));
        assert_eq!(output.matches(", ").count(), 639); // 确保有 639 个逗号
    }

    #[test]
    fn test_gasd_signature_512() {
        // 测试 GASDSignature512 的 Display 实现
        let gasd = GASDSignature512 { gasd: [3.0; 512] };
        let output = format!("{}", gasd);
        assert!(output.starts_with("(") && output.ends_with(")"));
        assert_eq!(output.matches(", ").count(), 511); // 确保有 511 个逗号
    }

    #[test]
    fn test_brisk_signature_512() {
        // 测试 BRISKSignature512 的 Display 实现
        let brisk = BRISKSignature512 {
            scale: 1.0,
            orientation: 0.5,
            descriptor: [0; 64],
        };
        let output = format!("{}", brisk);
        assert!(output.starts_with("1 0.5 (") && output.ends_with(")"));
        assert_eq!(output.matches(", ").count(), 63); // 确保有 63 个逗号
    }

    #[test]
    fn test_narf36() {
        // 测试 Narf36 的 Display 实现
        let narf = Narf36 {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            descriptor: [0.0; 36],
        };
        let output = format!("{}", narf);
        assert!(output.starts_with("1,2,3 - 0deg,0deg,0deg - (") && output.ends_with(")"));
        assert_eq!(output.matches(", ").count(), 35); // 确保有 40 个逗号
    }

    #[test]
    fn test_border_description() {
        // 测试 BorderDescription 的 Display 实现
        let border = BorderDescription {
            x: 10,
            y: 20,
            traits: [false; 32],
        };
        let output = format!("{}", border);
        assert_eq!(output, "(10,20)"); // 确保输出格式正确
    }
}

#[cfg(test)]
mod tests6 {
    use super::*;

    #[test]
    fn test_intensity_gradient() {
        // 测试 IntensityGradient 的 Display 实现
        let gradient = IntensityGradient {
            gradient: [1.0, 2.0, 3.0],
        };
        let output = format!("{}", gradient);
        assert_eq!(output, "(1,2,3)"); // 确保输出格式正确
    }

    #[test]
    fn test_histogram() {
        // 测试 Histogram 的 Display 实现
        let histogram = Histogram {
            histogram: [1.0, 2.0, 3.0, 4.0],
        };
        let output = format!("{}", histogram);
        assert!(output.starts_with("(") && output.ends_with(")"));
        assert_eq!(output.matches(", ").count(), 3); // 确保有 3 个逗号
    }

    #[test]
    fn test_point_with_scale() {
        // 测试 PointWithScale 的 Display 实现
        let point = PointWithScale {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            scale: 4.0,
            angle: 5.0,
            response: 6.0,
            octave: 7.0,
        };
        let output = format!("{}", point);
        assert_eq!(output, "(1,2,3 - 4,5,6,7)"); // 确保输出格式正确
    }

    #[test]
    fn test_point_surfel() {
        // 测试 PointSurfel 的 Display 实现
        let point = PointSurfel {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            normal: [4.0, 5.0, 6.0],
            rgba: 0xFFAABBCC,
            radius: 7.0,
            confidence: 8.0,
            curvature: 9.0,
        };
        let output = format!("{}", point);
        assert!(output.starts_with("(1,2,3 - 4,5,6 - 255,170,187,204 - 7 - 8 - 9)"));
        // 确保输出格式正确
    }

    #[test]
    fn test_point_dem() {
        // 测试 PointDEM 的 Display 实现
        let point = PointDEM {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            intensity: 4.0,
            intensity_variance: 5.0,
            height_variance: 6.0,
        };
        let output = format!("{}", point);
        assert_eq!(output, "(1,2,3 - 4 - 5 - 6)"); // 确保输出格式正确
    }
}
