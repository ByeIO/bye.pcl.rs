#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]

//! 实现点构造函数,
//! 为了安全性, 移除了内存对齐相关逻辑

extern crate quote;
extern crate syn;

// 线性代数
use nalgebra::{Const, DimName, storage::Storage};

/// 不能使用path方式引入,否则会找不到impl实现
// #[path = "./defines.rs"]
// mod defines;
use super::defines::{
    Axis, BRISKSignature512, BorderDescription, BorderTraits, Boundary, CPPFSignature,
    ESFSignature640, FPFHSignature33, GASDSignature512, GASDSignature7992, GASDSignature984,
    GFPFHSignature16, GRSDSignature21, Histogram, Intensity, Intensity32u, Intensity8u,
    IntensityGradient, InterestPoint, Label, MomentInvariants, Narf36, Normal,
    NormalBasedSignature12, PFHRGBSignature250, PFHSignature125, PPFRGBSignature, PPFSignature,
    PointDEM, PointNormal, PointSurfel, PointUV, PointWithRange, PointWithScale,
    PointWithViewpoint, PointXY, PointXYZ, PointXYZHSV, PointXYZI, PointXYZINormal, PointXYZL,
    PointXYZLAB, PointXYZLNormal, PointXYZRGB, PointXYZRGBA, PointXYZRGBL, PointXYZRGBNormal,
    PrincipalCurvatures, PrincipalRadiiRSD, ReferenceFrame, ShapeContext1980,
    UniqueShapeContext1960, VFHSignature308, PCL_DESCRIPTOR_FEATURE_POINT_TYPES,
    PCL_FEATURE_POINT_TYPES, PCL_NORMAL_POINT_TYPES, PCL_XYZL_POINT_TYPES, RGB, SHOT1344, SHOT352,
};

// #[path = "./descriptor_size.rs"]
// mod descriptor_size;
use super::descriptor_size;

// #[path = "./eigen_map.rs"]
// mod eigen_map;
use super::eigen_map::PclAddPoint4D;

/* start 点云构造函数 */

// 1. 实现PointXYZ构造函数
impl PointXYZ {
    // 默认构造函数，初始化为(0.0, 0.0, 0.0)
    pub const fn new() -> Self {
        PointXYZ {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    // 带参数的构造函数，初始化指定坐标
    pub const fn from_xyz(x: f32, y: f32, z: f32) -> Self {
        PointXYZ { x, y, z }
    }
}

// 2. 实现RGB的构造函数
impl RGB {
    // 默认构造函数，初始化为黑色
    pub const fn new() -> Self {
        RGB {
            r: 0,
            g: 0,
            b: 0,
            a: 255,
        }
    }

    // 带参数的构造函数，初始化指定颜色
    pub const fn from_rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        RGB { r, g, b, a }
    }
}

// 3. 实现Intensity(灰度强度)的构造函数
impl Intensity {
    // 默认构造函数，初始化为0.0
    pub const fn new() -> Self {
        Intensity { intensity: 0.0 }
    }

    // 带参数的构造函数，初始化指定强度
    pub const fn from_value(intensity: f32) -> Self {
        Intensity { intensity }
    }
}

// 4. 实现Intensity8u的构造函数
impl Intensity8u {
    // 默认构造函数，初始化为0
    pub const fn new() -> Self {
        Intensity8u { intensity: 0 }
    }

    // 带参数的构造函数，初始化指定强度
    pub const fn from_value(intensity: u8) -> Self {
        Intensity8u { intensity }
    }
}

// 5. 实现Intensity32u的构造函数
impl Intensity32u {
    // 默认构造函数，初始化为0
    pub const fn new() -> Self {
        Intensity32u { intensity: 0 }
    }

    // 带参数的构造函数，初始化指定强度
    pub const fn from_value(intensity: u32) -> Self {
        Intensity32u { intensity }
    }
}

// 6. PointXYZI 结构体，表示欧几里得坐标 xyz 和强度值
impl PointXYZI {
    // 构造函数，初始化所有字段
    pub fn new(x: f32, y: f32, z: f32, intensity: f32) -> Self {
        Self { x, y, z, intensity }
    }

    // 默认构造函数，强度值为 0
    pub fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }
}

// 7. PointXYZL 结构体，表示欧几里得坐标 xyz 和标签
impl PointXYZL {
    // 构造函数，初始化所有字段
    pub fn new(x: f32, y: f32, z: f32, label: u32) -> Self {
        Self { x, y, z, label }
    }

    // 默认构造函数，强度值为 0
    pub fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0)
    }
}

// 8. Label 结构体，表示标签
impl Label {
    // 构造函数，初始化标签
    pub fn new(label: u32) -> Self {
        Self { label }
    }

    // 默认构造函数，标签为 0
    pub fn default() -> Self {
        Self::new(0)
    }
}

// 9. PointXYZRGBA 结构体，表示欧几里得坐标 xyz 和 RGBA 颜色
impl PointXYZRGBA {
    // 构造函数，初始化所有字段
    pub fn new(x: f32, y: f32, z: f32, rgba: u32) -> Self {
        Self { x, y, z, rgba }
    }

    // 默认构造函数，RGBA 为白色
    pub fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0xFFFFFFFF)
    }
}

// 10. PointXYZRGB 结构体，表示欧几里得坐标 xyz 和 RGB 颜色
impl PointXYZRGB {
    // 构造函数，初始化所有字段
    pub fn new(x: f32, y: f32, z: f32, rgb: u32) -> Self {
        Self { x, y, z, rgb }
    }

    // 默认构造函数，RGB 为黑色
    pub fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0x000000)
    }
}

// 11. PointXYZRGBL 结构体，表示欧几里得坐标 xyz、RGB 颜色和标签
impl PointXYZRGBL {
    // 构造函数，初始化所有字段
    pub fn new(x: f32, y: f32, z: f32, rgb: u32, label: u32) -> Self {
        Self {
            x,
            y,
            z,
            rgb,
            label,
        }
    }

    // 默认构造函数，RGB 为黑色，标签为 0
    pub fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0x000000, 0)
    }
}

// 12. 表示带有CIELAB颜色的欧几里得xyz坐标的点结构体
impl PointXYZLAB {
    // 构造函数：从_PointXYZLAB结构体初始化
    pub fn new_from_raw(p: &PointXYZLAB) -> Self {
        PointXYZLAB {
            x: p.x,
            y: p.y,
            z: p.z,
            l: p.l,
            a: p.a,
            b: p.b,
        }
    }

    // 默认构造函数
    pub fn new() -> Self {
        PointXYZLAB {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            l: 0.0,
            a: 0.0,
            b: 0.0,
        }
    }

    // 带参数构造函数
    pub fn new_with_values(x: f32, y: f32, z: f32, l: f32, a: f32, b: f32) -> Self {
        PointXYZLAB { x, y, z, l, a, b }
    }
}

// 13. 表示带有HSV颜色的欧几里得xyz坐标的点结构体
impl PointXYZHSV {
    // 构造函数：从_PointXYZHSV结构体初始化
    pub fn new_from_raw(p: &PointXYZHSV) -> Self {
        PointXYZHSV {
            x: p.x,
            y: p.y,
            z: p.z,
            h: p.h,
            s: p.s,
            v: p.v,
        }
    }

    // 默认构造函数
    pub fn new() -> Self {
        PointXYZHSV {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            h: 0.0,
            s: 0.0,
            v: 0.0,
        }
    }

    // 带参数构造函数
    pub fn new_with_values(x: f32, y: f32, z: f32, h: f32, s: f32, v: f32) -> Self {
        PointXYZHSV { x, y, z, h, s, v }
    }
}

// 14. 表示欧几里得xy坐标的2D点结构体
impl PointXY {
    // 带参数构造函数
    pub fn new(x: f32, y: f32) -> Self {
        PointXY { x, y }
    }

    // 默认构造函数
    pub fn new_default() -> Self {
        PointXY { x: 0.0, y: 0.0 }
    }
}

// 15. 表示像素图像坐标的2D点结构体
impl PointUV {
    // 默认构造函数
    pub fn new() -> Self {
        PointUV { u: 0.0, v: 0.0 }
    }

    // 带参数构造函数
    pub fn new_with_values(u: f32, v: f32) -> Self {
        PointUV { u, v }
    }
}

// 16. 表示一个兴趣点，包含欧几里得xyz坐标和兴趣值
impl InterestPoint {
    // 构造函数，初始化所有字段
    pub fn new(x: f32, y: f32, z: f32, strength: f32) -> Self {
        Self { x, y, z, strength }
    }
}

// 17. 表示法线坐标和表面曲率估计
impl Normal {
    // 构造函数，初始化所有字段
    pub fn new(normal: [f32; 3], curvature: f32) -> Self {
        Self { normal, curvature }
    }

    // 构造函数，仅初始化曲率
    pub fn with_curvature(curvature: f32) -> Self {
        Self {
            normal: [0.0, 0.0, 0.0],
            curvature,
        }
    }
}

// 18. 表示一个轴，使用其法线坐标
impl Axis {
    // 构造函数，初始化所有字段
    pub fn new(normal: [f32; 3]) -> Self {
        Self { normal }
    }

    // 默认构造函数
    pub fn default() -> Self {
        Self {
            normal: [0.0, 0.0, 0.0],
        }
    }
}

// 19. 表示欧几里得xyz坐标，以及法线坐标和表面曲率估计
impl PointNormal {
    // 构造函数，初始化所有字段
    pub fn new(x: f32, y: f32, z: f32, normal: [f32; 3], curvature: f32) -> Self {
        Self {
            x,
            y,
            z,
            normal,
            curvature,
        }
    }

    // 构造函数，仅初始化曲率
    pub fn with_curvature(curvature: f32) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            normal: [0.0, 0.0, 0.0],
            curvature,
        }
    }
}

// 20. 表示欧几里得xyz坐标，RGB颜色，以及法线坐标和表面曲率估计
impl PointXYZRGBNormal {
    // 构造函数，初始化所有字段
    pub fn new(x: f32, y: f32, z: f32, rgb: u32, normal: [f32; 3], curvature: f32) -> Self {
        Self {
            x,
            y,
            z,
            rgb,
            normal,
            curvature,
        }
    }

    // 构造函数，仅初始化曲率
    pub fn with_curvature(curvature: f32) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            rgb: 0,
            normal: [0.0, 0.0, 0.0],
            curvature,
        }
    }
}

// 21. 表示带有强度、法线和曲率估计的欧几里得xyz坐标的点结构
impl PointXYZINormal {
    // 构造函数：从_PointXYZINormal结构体初始化
    pub fn new_from_base(p: &PointXYZINormal) -> Self {
        PointXYZINormal {
            x: p.x,
            y: p.y,
            z: p.z,
            intensity: p.intensity,
            normal: p.normal,
            curvature: p.curvature,
        }
    }

    // 构造函数：仅初始化强度
    pub fn new_with_intensity(intensity: f32) -> Self {
        PointXYZINormal {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            intensity,
            normal: [0.0; 3],
            curvature: 0.0,
        }
    }

    // 构造函数：初始化xyz坐标和强度
    pub fn new_with_xyz_intensity(x: f32, y: f32, z: f32, intensity: f32) -> Self {
        PointXYZINormal {
            x,
            y,
            z,
            intensity,
            normal: [0.0; 3],
            curvature: 0.0,
        }
    }

    // 构造函数：初始化所有字段
    pub fn new_with_all_fields(
        x: f32,
        y: f32,
        z: f32,
        intensity: f32,
        normal_x: f32,
        normal_y: f32,
        normal_z: f32,
        curvature: f32,
    ) -> Self {
        PointXYZINormal {
            x,
            y,
            z,
            intensity,
            normal: [normal_x, normal_y, normal_z],
            curvature,
        }
    }
}

// 22. 表示带有标签、法线和曲率估计的欧几里得xyz坐标的点结构
impl PointXYZLNormal {
    // 构造函数：从_PointXYZLNormal结构体初始化
    pub fn new_from_base(p: &PointXYZLNormal) -> Self {
        PointXYZLNormal {
            x: p.x,
            y: p.y,
            z: p.z,
            label: p.label,
            normal: p.normal,
            curvature: p.curvature,
        }
    }

    // 构造函数：仅初始化标签
    pub fn new_with_label(label: u32) -> Self {
        PointXYZLNormal {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            label,
            normal: [0.0; 3],
            curvature: 0.0,
        }
    }

    // 构造函数：初始化xyz坐标和标签
    pub fn new_with_xyz_label(x: f32, y: f32, z: f32, label: u32) -> Self {
        PointXYZLNormal {
            x,
            y,
            z,
            label,
            normal: [0.0; 3],
            curvature: 0.0,
        }
    }

    // 构造函数：初始化所有字段
    pub fn new_with_all_fields(
        x: f32,
        y: f32,
        z: f32,
        label: u32,
        normal_x: f32,
        normal_y: f32,
        normal_z: f32,
        curvature: f32,
    ) -> Self {
        PointXYZLNormal {
            x,
            y,
            z,
            label,
            normal: [normal_x, normal_y, normal_z],
            curvature,
        }
    }
}

// 23. 表示带有范围的欧几里得xyz坐标的点结构
impl PointWithRange {
    // 构造函数：从_PointWithRange结构体初始化
    pub fn new_from_base(p: &PointWithRange) -> Self {
        PointWithRange {
            point: p.point,
            range: p.range,
        }
    }

    // 构造函数：仅初始化范围
    pub fn new_with_range(range: f32) -> Self {
        PointWithRange {
            point: [0.0, 0.0, 0.0, 1.0],
            range,
        }
    }

    // 构造函数：初始化xyz坐标和范围
    pub fn new_with_xyz_range(x: f32, y: f32, z: f32, range: f32) -> Self {
        PointWithRange {
            point: [x, y, z, 1.0],
            range,
        }
    }
}

// 24. 表示带有视点的欧几里得xyz坐标的点结构
impl PointWithViewpoint {
    // 构造函数：从_PointWithViewpoint结构体初始化
    pub fn new_from_base(p: &PointWithViewpoint) -> Self {
        PointWithViewpoint {
            x: p.x,
            y: p.y,
            z: p.z,
            vp_x: p.vp_x,
            vp_y: p.vp_y,
            vp_z: p.vp_z,
        }
    }

    // 构造函数：默认初始化
    pub fn new() -> Self {
        PointWithViewpoint {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            vp_x: 0.0,
            vp_y: 0.0,
            vp_z: 0.0,
        }
    }

    // 构造函数：初始化xyz坐标
    pub fn new_with_xyz(x: f32, y: f32, z: f32) -> Self {
        PointWithViewpoint {
            x,
            y,
            z,
            vp_x: 0.0,
            vp_y: 0.0,
            vp_z: 0.0,
        }
    }

    // 构造函数：初始化所有字段
    pub fn new_with_all_fields(x: f32, y: f32, z: f32, vp_x: f32, vp_y: f32, vp_z: f32) -> Self {
        PointWithViewpoint {
            x,
            y,
            z,
            vp_x,
            vp_y,
            vp_z,
        }
    }
}

// 25. 表示三个矩不变量的点结构
impl MomentInvariants {
    // 构造函数：默认初始化
    pub fn new() -> Self {
        MomentInvariants {
            j1: 0.0,
            j2: 0.0,
            j3: 0.0,
        }
    }

    // 构造函数：初始化所有字段
    pub fn new_with_all_fields(j1: f32, j2: f32, j3: f32) -> Self {
        MomentInvariants { j1, j2, j3 }
    }
}

// 26. 成员: float r_min, r_max
impl PrincipalRadiiRSD {
    // 默认构造函数
    pub fn new() -> Self {
        Self {
            r_min: 0.0,
            r_max: 0.0,
        }
    }

    // 带参数构造函数
    pub fn with_values(r_min: f32, r_max: f32) -> Self {
        Self { r_min, r_max }
    }
}

// 27. 成员: u8 boundary_point
impl Boundary {
    // 默认构造函数
    pub fn new() -> Self {
        Self { boundary_point: 0 }
    }

    // 带参数构造函数
    pub fn with_value(boundary_point: u8) -> Self {
        Self { boundary_point }
    }
}

// 28. 成员: float principal_curvature[3], pc1, pc2
impl PrincipalCurvatures {
    // 默认构造函数
    pub fn new() -> Self {
        Self {
            principal_curvature: [0.0; 3],
            pc1: 0.0,
            pc2: 0.0,
        }
    }

    // 带参数构造函数
    pub fn with_values(x: f32, y: f32, z: f32, pc1: f32, pc2: f32) -> Self {
        Self {
            principal_curvature: [x, y, z],
            pc1,
            pc2,
        }
    }
}

// 29. 成员: float descriptor[352], rf[9]
impl SHOT352 {
    // 默认构造函数
    pub fn new() -> Self {
        Self {
            descriptor: [0.0; 352],
            rf: [0.0; 9],
        }
    }
}

// 30. 成员: float descriptor[1344], rf[9]
impl SHOT1344 {
    // 默认构造函数
    pub fn new() -> Self {
        Self {
            descriptor: [0.0; 1344],
            rf: [0.0; 9],
        }
    }
}

// 31. 表示点的局部参考系
impl ReferenceFrame {
    // 默认构造函数，初始化所有轴为0
    pub fn new() -> Self {
        Self {
            x_axis: Axis { normal: [0.0; 3] },
            y_axis: Axis { normal: [0.0; 3] },
            z_axis: Axis { normal: [0.0; 3] },
        }
    }

    // 从数组初始化参考系
    pub fn from_rf(rf: [f32; 9]) -> Self {
        Self {
            x_axis: Axis {
                normal: [rf[0], rf[1], rf[2]],
            },
            y_axis: Axis {
                normal: [rf[3], rf[4], rf[5]],
            },
            z_axis: Axis {
                normal: [rf[6], rf[7], rf[8]],
            },
        }
    }
}

// 32. 表示Shape Context特征
impl ShapeContext1980 {
    // 默认构造函数，初始化所有值为0
    pub fn new() -> Self {
        Self {
            descriptor: [0.0; 1980],
            rf: [0.0; 9],
        }
    }
}

// 33. 表示Unique Shape Context特征
impl UniqueShapeContext1960 {
    // 默认构造函数，初始化所有值为0
    pub fn new() -> Self {
        Self {
            descriptor: [0.0; 1960],
            rf: [0.0; 9],
        }
    }
}

// 34. 表示PFH特征
impl PFHSignature125 {
    // 默认构造函数，初始化所有值为0
    pub fn new() -> Self {
        Self { pfh: [0.0; 125] }
    }
}

// 35. 表示PFHRGB特征
impl PFHRGBSignature250 {
    // 默认构造函数，初始化所有值为0
    pub fn new() -> Self {
        Self { pfhrgb: [0.0; 250] }
    }
}

// 36. PPFSignature 结构体
impl PPFSignature {
    // 构造函数1: 仅初始化 alpha_m
    pub fn new(alpha: f32) -> Self {
        Self {
            f1: 0.0,
            f2: 0.0,
            f3: 0.0,
            f4: 0.0,
            alpha_m: alpha,
        }
    }

    // 构造函数2: 初始化所有字段
    pub fn with_values(f1: f32, f2: f32, f3: f32, f4: f32, alpha: f32) -> Self {
        Self {
            f1,
            f2,
            f3,
            f4,
            alpha_m: alpha,
        }
    }
}

// 37. CPPFSignature 结构体
impl CPPFSignature {
    // 构造函数1: 仅初始化 alpha_m
    pub fn new(alpha: f32) -> Self {
        Self {
            f1: 0.0,
            f2: 0.0,
            f3: 0.0,
            f4: 0.0,
            f5: 0.0,
            f6: 0.0,
            f7: 0.0,
            f8: 0.0,
            f9: 0.0,
            f10: 0.0,
            alpha_m: alpha,
        }
    }

    // 构造函数2: 初始化所有字段
    pub fn with_values(
        f1: f32,
        f2: f32,
        f3: f32,
        f4: f32,
        f5: f32,
        f6: f32,
        f7: f32,
        f8: f32,
        f9: f32,
        f10: f32,
        alpha: f32,
    ) -> Self {
        Self {
            f1,
            f2,
            f3,
            f4,
            f5,
            f6,
            f7,
            f8,
            f9,
            f10,
            alpha_m: alpha,
        }
    }
}

// 38. PPFRGBSignature 结构体
impl PPFRGBSignature {
    // 构造函数1: 仅初始化 alpha_m
    pub fn new(alpha: f32) -> Self {
        Self {
            f1: 0.0,
            f2: 0.0,
            f3: 0.0,
            f4: 0.0,
            r_ratio: 0.0,
            g_ratio: 0.0,
            b_ratio: 0.0,
            alpha_m: alpha,
        }
    }

    // 构造函数2: 初始化部分字段
    pub fn with_values(f1: f32, f2: f32, f3: f32, f4: f32, alpha: f32) -> Self {
        Self {
            f1,
            f2,
            f3,
            f4,
            r_ratio: 0.0,
            g_ratio: 0.0,
            b_ratio: 0.0,
            alpha_m: alpha,
        }
    }

    // 构造函数3: 初始化所有字段
    pub fn with_all_values(
        f1: f32,
        f2: f32,
        f3: f32,
        f4: f32,
        alpha: f32,
        r: f32,
        g: f32,
        b: f32,
    ) -> Self {
        Self {
            f1,
            f2,
            f3,
            f4,
            r_ratio: r,
            g_ratio: g,
            b_ratio: b,
            alpha_m: alpha,
        }
    }
}

// 39. NormalBasedSignature12 结构体
impl NormalBasedSignature12 {
    // 默认构造函数
    pub fn new() -> Self {
        Self { values: [0.0; 12] }
    }
}

// 40. FPFHSignature33 结构体
impl FPFHSignature33 {
    // 默认构造函数
    pub fn new() -> Self {
        Self { fpfh: [0.0; 33] }
    }

    // 获取描述符大小
    pub fn descriptor_size() -> usize {
        33
    }
}

// 41. 表示视点特征直方图(VFH)的点结构
impl VFHSignature308 {
    // 获取描述符大小
    pub const fn descriptor_size() -> usize {
        308
    }

    // 默认构造函数
    pub const fn new() -> Self {
        Self { vfh: [0.0; 308] }
    }
}

// 42. 表示基于全局半径的表面描述符(GRSD)的点结构
impl GRSDSignature21 {
    pub const fn descriptor_size() -> usize {
        21
    }

    pub const fn new() -> Self {
        Self { grsd: [0.0; 21] }
    }
}

// 43. 表示形状函数集合(ESF)的点结构
impl ESFSignature640 {
    pub const fn descriptor_size() -> usize {
        640
    }

    pub const fn new() -> Self {
        Self { esf: [0.0; 640] }
    }
}

// 44. 表示全局对齐空间分布(GASD)形状描述符的点结构
impl GASDSignature512 {
    pub const fn descriptor_size() -> usize {
        512
    }

    pub const fn new() -> Self {
        Self { gasd: [0.0; 512] }
    }
}

// 45. 表示全局对齐空间分布(GASD)形状和颜色描述符的点结构
impl GASDSignature984 {
    pub const fn descriptor_size() -> usize {
        984
    }

    pub const fn new() -> Self {
        Self { gasd: [0.0; 984] }
    }
}

// 46. GASD描述符，包含7992个浮点数的直方图
impl GASDSignature7992 {
    // 默认构造函数
    pub fn new() -> Self {
        Self { gasd: [0.0; 7992] }
    }
}

// 47. GFPFH描述符，包含16个浮点数的直方图
impl GFPFHSignature16 {
    // 默认构造函数
    pub fn new() -> Self {
        Self { gfpfh: [0.0; 16] }
    }
}

// 48. BRISK描述符，包含尺度和方向信息，以及64字节的描述符
impl BRISKSignature512 {
    // 默认构造函数
    pub fn new() -> Self {
        Self {
            scale: 0.0,
            orientation: 0.0,
            descriptor: [0; 64],
        }
    }

    // 带参数的构造函数
    pub fn with_params(scale: f32, orientation: f32) -> Self {
        Self {
            scale,
            orientation,
            descriptor: [0; 64],
        }
    }
}

// 49. Narf描述符，包含位置、姿态信息和36个浮点数的描述符
impl Narf36 {
    // 默认构造函数
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            descriptor: [0.0; 36],
        }
    }

    // 带位置参数的构造函数
    pub fn with_position(x: f32, y: f32, z: f32) -> Self {
        Self {
            x,
            y,
            z,
            roll: 0.0,
            pitch: 0.0,
            yaw: 0.0,
            descriptor: [0.0; 36],
        }
    }

    // 带完整参数的构造函数
    pub fn with_all_params(x: f32, y: f32, z: f32, roll: f32, pitch: f32, yaw: f32) -> Self {
        Self {
            x,
            y,
            z,
            roll,
            pitch,
            yaw,
            descriptor: [0.0; 36],
        }
    }
}

// 50. 边界描述符，包含坐标和32个布尔值的特征
impl BorderDescription {
    // 默认构造函数
    pub fn new() -> Self {
        Self {
            x: 0,
            y: 0,
            traits: [false; 32],
        }
    }

    // 带坐标参数的构造函数
    pub fn with_position(x: i32, y: i32) -> Self {
        Self {
            x,
            y,
            traits: [false; 32],
        }
    }
}

// 51. 强度梯度结构体
impl IntensityGradient {
    // 默认构造函数
    pub fn new() -> Self {
        Self {
            gradient: [0.0, 0.0, 0.0],
        }
    }

    // 带参数构造函数
    pub fn with_values(x: f32, y: f32, z: f32) -> Self {
        Self {
            gradient: [x, y, z],
        }
    }
}

// 52. 直方图结构体
impl<const N: usize> Histogram<N> {
    // 获取描述符大小
    pub fn descriptor_size() -> usize {
        N
    }
}

// 53. 带尺度的点结构体
impl PointWithScale {
    // 默认构造函数
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            scale: 1.0,
            angle: -1.0,
            response: 0.0,
            octave: 0.0,
        }
    }

    // 带参数构造函数
    pub fn with_values(
        x: f32,
        y: f32,
        z: f32,
        scale: f32,
        angle: f32,
        response: f32,
        octave: f32,
    ) -> Self {
        Self {
            x,
            y,
            z,
            scale,
            angle,
            response,
            octave,
        }
    }
}

// 54. 表面点结构体
impl PointSurfel {
    // 默认构造函数
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            normal: [0.0, 0.0, 0.0],
            rgba: 0,
            radius: 0.0,
            confidence: 0.0,
            curvature: 0.0,
        }
    }

    // 带参数构造函数
    pub fn with_values(
        x: f32,
        y: f32,
        z: f32,
        nx: f32,
        ny: f32,
        nz: f32,
        r: u8,
        g: u8,
        b: u8,
        a: u8,
        radius: f32,
        confidence: f32,
        curvature: f32,
    ) -> Self {
        Self {
            x,
            y,
            z,
            normal: [nx, ny, nz],
            rgba: ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | (a as u32),
            radius,
            confidence,
            curvature,
        }
    }
}

// 55. 数字高程图点结构体
impl PointDEM {
    // 默认构造函数
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            intensity: 0.0,
            intensity_variance: 0.0,
            height_variance: 0.0,
        }
    }

    // 带参数构造函数
    pub fn with_values(
        x: f32,
        y: f32,
        z: f32,
        intensity: f32,
        intensity_variance: f32,
        height_variance: f32,
    ) -> Self {
        Self {
            x,
            y,
            z,
            intensity,
            intensity_variance,
            height_variance,
        }
    }
}

/* end 点云构造函数 */

#[cfg(test)]
mod tests1 {
    use super::*;

    // 1. 测试PointXYZ的构造函数
    #[test]
    fn test_point_xyz() {
        let p1 = PointXYZ::new();
        assert_eq!(p1.x, 0.0);
        assert_eq!(p1.y, 0.0);
        assert_eq!(p1.z, 0.0);

        let p2 = PointXYZ::from_xyz(1.0, 2.0, 3.0);
        assert_eq!(p2.x, 1.0);
        assert_eq!(p2.y, 2.0);
        assert_eq!(p2.z, 3.0);
    }

    // 2. 测试RGB的构造函数
    #[test]
    fn test_rgb() {
        let c1 = RGB::new();
        assert_eq!(c1.r, 0);
        assert_eq!(c1.g, 0);
        assert_eq!(c1.b, 0);
        assert_eq!(c1.a, 255);

        let c2 = RGB::from_rgba(255, 128, 64, 200);
        assert_eq!(c2.r, 255);
        assert_eq!(c2.g, 128);
        assert_eq!(c2.b, 64);
        assert_eq!(c2.a, 200);
    }

    // 3. 测试Intensity的构造函数
    #[test]
    fn test_intensity() {
        let i1 = Intensity::new();
        assert_eq!(i1.intensity, 0.0);

        let i2 = Intensity::from_value(0.5);
        assert_eq!(i2.intensity, 0.5);
    }

    // 4. 测试Intensity8u的构造函数
    #[test]
    fn test_intensity8u() {
        let i1 = Intensity8u::new();
        assert_eq!(i1.intensity, 0);

        let i2 = Intensity8u::from_value(128);
        assert_eq!(i2.intensity, 128);
    }

    // 5. 测试Intensity32u的构造函数
    #[test]
    fn test_intensity32u() {
        let i1 = Intensity32u::new();
        assert_eq!(i1.intensity, 0);

        let i2 = Intensity32u::from_value(4294967295);
        assert_eq!(i2.intensity, 4294967295);
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;

    // 测试 PointXYZI 结构体的构造函数和默认构造函数
    #[test]
    fn test_point_xyzi() {
        let point = PointXYZI::new(1.0, 2.0, 3.0, 0.5);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.intensity, 0.5);

        let default_point = PointXYZI::default();
        assert_eq!(default_point.x, 0.0);
        assert_eq!(default_point.y, 0.0);
        assert_eq!(default_point.z, 0.0);
        assert_eq!(default_point.intensity, 0.0);
    }

    // 测试 PointXYZL 结构体的构造函数和默认构造函数
    #[test]
    fn test_point_xyzl() {
        let point = PointXYZL::new(1.0, 2.0, 3.0, 10);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.label, 10);

        let default_point = PointXYZL::default();
        assert_eq!(default_point.x, 0.0);
        assert_eq!(default_point.y, 0.0);
        assert_eq!(default_point.z, 0.0);
        assert_eq!(default_point.label, 0);
    }

    // 测试 Label 结构体的构造函数和默认构造函数
    #[test]
    fn test_label() {
        let label = Label::new(5);
        assert_eq!(label.label, 5);

        let default_label = Label::default();
        assert_eq!(default_label.label, 0);
    }

    // 测试 PointXYZRGBA 结构体的构造函数和默认构造函数
    #[test]
    fn test_point_xyzrgba() {
        let point = PointXYZRGBA::new(1.0, 2.0, 3.0, 0xFF0000FF);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.rgba, 0xFF0000FF);

        let default_point = PointXYZRGBA::default();
        assert_eq!(default_point.x, 0.0);
        assert_eq!(default_point.y, 0.0);
        assert_eq!(default_point.z, 0.0);
        assert_eq!(default_point.rgba, 0xFFFFFFFF);
    }

    // 测试 PointXYZRGB 结构体的构造函数和默认构造函数
    #[test]
    fn test_point_xyzrgb() {
        let point = PointXYZRGB::new(1.0, 2.0, 3.0, 0x00FF00);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.rgb, 0x00FF00);

        let default_point = PointXYZRGB::default();
        assert_eq!(default_point.x, 0.0);
        assert_eq!(default_point.y, 0.0);
        assert_eq!(default_point.z, 0.0);
        assert_eq!(default_point.rgb, 0x000000);
    }
}

#[cfg(test)]
mod tests3 {
    use super::*;
    // 1. 测试 PointXYZRGBL 结构体
    #[test]
    fn test_point_xyzrgb_l() {
        // 测试默认构造函数
        let default_point = PointXYZRGBL::default();
        assert_eq!(default_point.x, 0.0);
        assert_eq!(default_point.y, 0.0);
        assert_eq!(default_point.z, 0.0);
        assert_eq!(default_point.rgb, 0x000000);
        assert_eq!(default_point.label, 0);

        // 测试带参数构造函数
        let custom_point = PointXYZRGBL::new(1.0, 2.0, 3.0, 0xFF0000, 1);
        assert_eq!(custom_point.x, 1.0);
        assert_eq!(custom_point.y, 2.0);
        assert_eq!(custom_point.z, 3.0);
        assert_eq!(custom_point.rgb, 0xFF0000);
        assert_eq!(custom_point.label, 1);
    }

    // 2. 测试 PointXYZLAB 结构体
    #[test]
    fn test_point_xyzlab() {
        // 测试默认构造函数
        let default_point = PointXYZLAB::new();
        assert_eq!(default_point.x, 0.0);
        assert_eq!(default_point.y, 0.0);
        assert_eq!(default_point.z, 0.0);
        assert_eq!(default_point.l, 0.0);
        assert_eq!(default_point.a, 0.0);
        assert_eq!(default_point.b, 0.0);

        // 测试带参数构造函数
        let custom_point = PointXYZLAB::new_with_values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(custom_point.x, 1.0);
        assert_eq!(custom_point.y, 2.0);
        assert_eq!(custom_point.z, 3.0);
        assert_eq!(custom_point.l, 4.0);
        assert_eq!(custom_point.a, 5.0);
        assert_eq!(custom_point.b, 6.0);

        // 测试从原始结构体初始化
        let raw_point = PointXYZLAB {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            l: 4.0,
            a: 5.0,
            b: 6.0,
        };
        let new_point = PointXYZLAB::new_from_raw(&raw_point);
        assert_eq!(new_point.x, 1.0);
        assert_eq!(new_point.y, 2.0);
        assert_eq!(new_point.z, 3.0);
        assert_eq!(new_point.l, 4.0);
        assert_eq!(new_point.a, 5.0);
        assert_eq!(new_point.b, 6.0);
    }

    // 3. 测试 PointXYZHSV 结构体
    #[test]
    fn test_point_xyzhsv() {
        // 测试默认构造函数
        let default_point = PointXYZHSV::new();
        assert_eq!(default_point.x, 0.0);
        assert_eq!(default_point.y, 0.0);
        assert_eq!(default_point.z, 0.0);
        assert_eq!(default_point.h, 0.0);
        assert_eq!(default_point.s, 0.0);
        assert_eq!(default_point.v, 0.0);

        // 测试带参数构造函数
        let custom_point = PointXYZHSV::new_with_values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(custom_point.x, 1.0);
        assert_eq!(custom_point.y, 2.0);
        assert_eq!(custom_point.z, 3.0);
        assert_eq!(custom_point.h, 4.0);
        assert_eq!(custom_point.s, 5.0);
        assert_eq!(custom_point.v, 6.0);

        // 测试从原始结构体初始化
        let raw_point = PointXYZHSV {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            h: 4.0,
            s: 5.0,
            v: 6.0,
        };
        let new_point = PointXYZHSV::new_from_raw(&raw_point);
        assert_eq!(new_point.x, 1.0);
        assert_eq!(new_point.y, 2.0);
        assert_eq!(new_point.z, 3.0);
        assert_eq!(new_point.h, 4.0);
        assert_eq!(new_point.s, 5.0);
        assert_eq!(new_point.v, 6.0);
    }

    // 4. 测试 PointXY 结构体
    #[test]
    fn test_point_xy() {
        // 测试默认构造函数
        let default_point = PointXY::new_default();
        assert_eq!(default_point.x, 0.0);
        assert_eq!(default_point.y, 0.0);

        // 测试带参数构造函数
        let custom_point = PointXY::new(1.0, 2.0);
        assert_eq!(custom_point.x, 1.0);
        assert_eq!(custom_point.y, 2.0);
    }

    // 5. 测试 PointUV 结构体
    #[test]
    fn test_point_uv() {
        // 测试默认构造函数
        let default_point = PointUV::new();
        assert_eq!(default_point.u, 0.0);
        assert_eq!(default_point.v, 0.0);

        // 测试带参数构造函数
        let custom_point = PointUV::new_with_values(1.0, 2.0);
        assert_eq!(custom_point.u, 1.0);
        assert_eq!(custom_point.v, 2.0);
    }
}

#[cfg(test)]
mod tests4 {
    use super::*;
    // 1. 测试InterestPoint的构造函数
    #[test]
    fn test_interest_point_new() {
        let point = InterestPoint::new(1.0, 2.0, 3.0, 0.5);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.strength, 0.5);
    }

    // 2. 测试Normal的构造函数
    #[test]
    fn test_normal_new() {
        let normal = Normal::new([1.0, 0.0, 0.0], 0.1);
        assert_eq!(normal.normal, [1.0, 0.0, 0.0]);
        assert_eq!(normal.curvature, 0.1);
    }

    // 3. 测试Normal的with_curvature构造函数
    #[test]
    fn test_normal_with_curvature() {
        let normal = Normal::with_curvature(0.2);
        assert_eq!(normal.normal, [0.0, 0.0, 0.0]);
        assert_eq!(normal.curvature, 0.2);
    }

    // 4. 测试Axis的构造函数
    #[test]
    fn test_axis_new() {
        let axis = Axis::new([0.0, 1.0, 0.0]);
        assert_eq!(axis.normal, [0.0, 1.0, 0.0]);
    }

    // 5. 测试Axis的默认构造函数
    #[test]
    fn test_axis_default() {
        let axis = Axis::default();
        assert_eq!(axis.normal, [0.0, 0.0, 0.0]);
    }

    // 6. 测试PointNormal的构造函数
    #[test]
    fn test_point_normal_new() {
        let point = PointNormal::new(1.0, 2.0, 3.0, [0.0, 0.0, 1.0], 0.3);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.normal, [0.0, 0.0, 1.0]);
        assert_eq!(point.curvature, 0.3);
    }

    // 7. 测试PointNormal的with_curvature构造函数
    #[test]
    fn test_point_normal_with_curvature() {
        let point = PointNormal::with_curvature(0.4);
        assert_eq!(point.x, 0.0);
        assert_eq!(point.y, 0.0);
        assert_eq!(point.z, 0.0);
        assert_eq!(point.normal, [0.0, 0.0, 0.0]);
        assert_eq!(point.curvature, 0.4);
    }

    // 8. 测试PointXYZRGBNormal的构造函数
    #[test]
    fn test_point_xyz_rgb_normal_new() {
        let point = PointXYZRGBNormal::new(1.0, 2.0, 3.0, 0xFF0000, [0.0, 1.0, 0.0], 0.5);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.rgb, 0xFF0000);
        assert_eq!(point.normal, [0.0, 1.0, 0.0]);
        assert_eq!(point.curvature, 0.5);
    }

    // 9. 测试PointXYZRGBNormal的with_curvature构造函数
    #[test]
    fn test_point_xyz_rgb_normal_with_curvature() {
        let point = PointXYZRGBNormal::with_curvature(0.6);
        assert_eq!(point.x, 0.0);
        assert_eq!(point.y, 0.0);
        assert_eq!(point.z, 0.0);
        assert_eq!(point.rgb, 0);
        assert_eq!(point.normal, [0.0, 0.0, 0.0]);
        assert_eq!(point.curvature, 0.6);
    }
}

mod test5 {
    use super::*;

    // 1. 测试PointXYZINormal的构造函数
    #[test]
    fn test_point_xyz_i_normal() {
        // 1.1 测试从基础结构体初始化
        let base_point = PointXYZINormal {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            intensity: 0.5,
            normal: [0.1, 0.2, 0.3],
            curvature: 0.01,
        };
        let point = PointXYZINormal::new_from_base(&base_point);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.intensity, 0.5);
        assert_eq!(point.normal, [0.1, 0.2, 0.3]);
        assert_eq!(point.curvature, 0.01);

        // 1.2 测试仅初始化强度
        let point = PointXYZINormal::new_with_intensity(0.8);
        assert_eq!(point.intensity, 0.8);
        assert_eq!(point.x, 0.0);
        assert_eq!(point.y, 0.0);
        assert_eq!(point.z, 0.0);

        // 1.3 测试初始化xyz坐标和强度
        let point = PointXYZINormal::new_with_xyz_intensity(1.0, 2.0, 3.0, 0.5);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.intensity, 0.5);

        // 1.4 测试初始化所有字段
        let point = PointXYZINormal::new_with_all_fields(1.0, 2.0, 3.0, 0.5, 0.1, 0.2, 0.3, 0.01);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.intensity, 0.5);
        assert_eq!(point.normal, [0.1, 0.2, 0.3]);
        assert_eq!(point.curvature, 0.01);
    }

    // 2. 测试PointXYZLNormal的构造函数
    #[test]
    fn test_point_xyz_l_normal() {
        // 2.1 测试从基础结构体初始化
        let base_point = PointXYZLNormal {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            label: 1,
            normal: [0.1, 0.2, 0.3],
            curvature: 0.01,
        };
        let point = PointXYZLNormal::new_from_base(&base_point);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.label, 1);
        assert_eq!(point.normal, [0.1, 0.2, 0.3]);
        assert_eq!(point.curvature, 0.01);

        // 2.2 测试仅初始化标签
        let point = PointXYZLNormal::new_with_label(2);
        assert_eq!(point.label, 2);
        assert_eq!(point.x, 0.0);
        assert_eq!(point.y, 0.0);
        assert_eq!(point.z, 0.0);

        // 2.3 测试初始化xyz坐标和标签
        let point = PointXYZLNormal::new_with_xyz_label(1.0, 2.0, 3.0, 3);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.label, 3);

        // 2.4 测试初始化所有字段
        let point = PointXYZLNormal::new_with_all_fields(1.0, 2.0, 3.0, 4, 0.1, 0.2, 0.3, 0.01);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.label, 4);
        assert_eq!(point.normal, [0.1, 0.2, 0.3]);
        assert_eq!(point.curvature, 0.01);
    }

    // 3. 测试PointWithRange的构造函数
    #[test]
    fn test_point_with_range() {
        // 3.1 测试从基础结构体初始化
        let base_point = PointWithRange {
            point: [1.0, 2.0, 3.0, 1.0],
            range: 10.0,
        };
        let point = PointWithRange::new_from_base(&base_point);
        assert_eq!(point.point, [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(point.range, 10.0);

        // 3.2 测试仅初始化范围
        let point = PointWithRange::new_with_range(15.0);
        assert_eq!(point.range, 15.0);
        assert_eq!(point.point, [0.0, 0.0, 0.0, 1.0]);

        // 3.3 测试初始化xyz坐标和范围
        let point = PointWithRange::new_with_xyz_range(1.0, 2.0, 3.0, 20.0);
        assert_eq!(point.point, [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(point.range, 20.0);
    }

    // 4. 测试PointWithViewpoint的构造函数
    #[test]
    fn test_point_with_viewpoint() {
        // 4.1 测试从基础结构体初始化
        let base_point = PointWithViewpoint {
            x: 1.0,
            y: 2.0,
            z: 3.0,
            vp_x: 4.0,
            vp_y: 5.0,
            vp_z: 6.0,
        };
        let point = PointWithViewpoint::new_from_base(&base_point);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.vp_x, 4.0);
        assert_eq!(point.vp_y, 5.0);
        assert_eq!(point.vp_z, 6.0);

        // 4.2 测试默认初始化
        let point = PointWithViewpoint::new();
        assert_eq!(point.x, 0.0);
        assert_eq!(point.y, 0.0);
        assert_eq!(point.z, 0.0);
        assert_eq!(point.vp_x, 0.0);
        assert_eq!(point.vp_y, 0.0);
        assert_eq!(point.vp_z, 0.0);

        // 4.3 测试初始化xyz坐标
        let point = PointWithViewpoint::new_with_xyz(1.0, 2.0, 3.0);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.vp_x, 0.0);
        assert_eq!(point.vp_y, 0.0);
        assert_eq!(point.vp_z, 0.0);

        // 4.4 测试初始化所有字段
        let point = PointWithViewpoint::new_with_all_fields(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);
        assert_eq!(point.vp_x, 4.0);
        assert_eq!(point.vp_y, 5.0);
        assert_eq!(point.vp_z, 6.0);
    }

    // 5. 测试MomentInvariants的构造函数
    #[test]
    fn test_moment_invariants() {
        // 5.1 测试默认初始化
        let moment = MomentInvariants::new();
        assert_eq!(moment.j1, 0.0);
        assert_eq!(moment.j2, 0.0);
        assert_eq!(moment.j3, 0.0);

        // 5.2 测试初始化所有字段
        let moment = MomentInvariants::new_with_all_fields(1.0, 2.0, 3.0);
        assert_eq!(moment.j1, 1.0);
        assert_eq!(moment.j2, 2.0);
        assert_eq!(moment.j3, 3.0);
    }
}

#[cfg(test)]
mod test6 {
    use super::*;

    #[test]
    fn test_principal_radii_rsd() {
        let p1 = PrincipalRadiiRSD::new();
        assert_eq!(p1.r_min, 0.0);
        assert_eq!(p1.r_max, 0.0);

        let p2 = PrincipalRadiiRSD::with_values(1.0, 2.0);
        assert_eq!(p2.r_min, 1.0);
        assert_eq!(p2.r_max, 2.0);
    }

    #[test]
    fn test_boundary() {
        let b1 = Boundary::new();
        assert_eq!(b1.boundary_point, 0);

        let b2 = Boundary::with_value(1);
        assert_eq!(b2.boundary_point, 1);
    }

    #[test]
    fn test_principal_curvatures() {
        let pc1 = PrincipalCurvatures::new();
        assert_eq!(pc1.principal_curvature, [0.0; 3]);
        assert_eq!(pc1.pc1, 0.0);
        assert_eq!(pc1.pc2, 0.0);

        let pc2 = PrincipalCurvatures::with_values(1.0, 2.0, 3.0, 4.0, 5.0);
        assert_eq!(pc2.principal_curvature, [1.0, 2.0, 3.0]);
        assert_eq!(pc2.pc1, 4.0);
        assert_eq!(pc2.pc2, 5.0);
    }

    #[test]
    fn test_shot352() {
        let s1 = SHOT352::new();
        assert_eq!(s1.descriptor.len(), 352);
        assert_eq!(s1.rf.len(), 9);
    }

    #[test]
    fn test_shot1344() {
        let s1 = SHOT1344::new();
        assert_eq!(s1.descriptor.len(), 1344);
        assert_eq!(s1.rf.len(), 9);
    }
}

#[cfg(test)]
mod test7 {
    use super::*;

    #[test]
    fn test_reference_frame() {
        let rf = ReferenceFrame::new();
        assert_eq!(
            rf.x_axis,
            Axis {
                normal: [0.0, 0.0, 0.0]
            }
        );
        assert_eq!(
            rf.y_axis,
            Axis {
                normal: [0.0, 0.0, 0.0]
            }
        );
        assert_eq!(
            rf.z_axis,
            Axis {
                normal: [0.0, 0.0, 0.0]
            }
        );

        let rf = ReferenceFrame::from_rf([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(
            rf.x_axis,
            Axis {
                normal: [1.0, 2.0, 3.0]
            }
        );
        assert_eq!(
            rf.y_axis,
            Axis {
                normal: [4.0, 5.0, 6.0]
            }
        );
        assert_eq!(
            rf.z_axis,
            Axis {
                normal: [7.0, 8.0, 9.0]
            }
        );
    }

    #[test]
    fn test_shape_context() {
        let sc = ShapeContext1980::new();
        assert_eq!(sc.descriptor[0], 0.0);
        assert_eq!(sc.descriptor[1979], 0.0);
        assert_eq!(sc.rf[0], 0.0);
        assert_eq!(sc.rf[8], 0.0);
    }

    #[test]
    fn test_unique_shape_context() {
        let usc = UniqueShapeContext1960::new();
        assert_eq!(usc.descriptor[0], 0.0);
        assert_eq!(usc.descriptor[1959], 0.0);
        assert_eq!(usc.rf[0], 0.0);
        assert_eq!(usc.rf[8], 0.0);
    }

    #[test]
    fn test_pfh_signature() {
        let pfh = PFHSignature125::new();
        assert_eq!(pfh.pfh[0], 0.0);
        assert_eq!(pfh.pfh[124], 0.0);
    }

    #[test]
    fn test_pfhrgb_signature() {
        let pfhrgb = PFHRGBSignature250::new();
        assert_eq!(pfhrgb.pfhrgb[0], 0.0);
        assert_eq!(pfhrgb.pfhrgb[249], 0.0);
    }
}

// 测试模块
#[cfg(test)]
mod test8 {
    use super::*;

    #[test]
    fn test_ppf_signature() {
        let ppf = PPFSignature::with_values(1.0, 2.0, 3.0, 4.0, 0.5);
        assert_eq!(ppf.f1, 1.0);
        assert_eq!(ppf.alpha_m, 0.5);
    }

    #[test]
    fn test_cppf_signature() {
        let cppf =
            CPPFSignature::with_values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 0.5);
        assert_eq!(cppf.f10, 10.0);
    }

    #[test]
    fn test_ppfrgb_signature() {
        let ppfrgb = PPFRGBSignature::with_all_values(1.0, 2.0, 3.0, 4.0, 0.5, 0.1, 0.2, 0.3);
        assert_eq!(ppfrgb.r_ratio, 0.1);
    }

    #[test]
    fn test_normal_based_signature12() {
        let normal = NormalBasedSignature12::new();
        assert_eq!(normal.values[0], 0.0);
    }

    #[test]
    fn test_fpfh_signature33() {
        let fpfh = FPFHSignature33::new();
        assert_eq!(fpfh.fpfh[0], 0.0);
        assert_eq!(FPFHSignature33::descriptor_size(), 33);
    }
}

#[cfg(test)]
mod test9 {
    use super::*;

    #[test]
    fn test_vfh_signature() {
        let vfh = VFHSignature308::new();
        assert_eq!(vfh.vfh.len(), 308);
        assert_eq!(VFHSignature308::descriptor_size(), 308);
    }

    #[test]
    fn test_grsd_signature() {
        let grsd = GRSDSignature21::new();
        assert_eq!(grsd.grsd.len(), 21);
        assert_eq!(GRSDSignature21::descriptor_size(), 21);
    }

    #[test]
    fn test_esf_signature() {
        let esf = ESFSignature640::new();
        assert_eq!(esf.esf.len(), 640);
        assert_eq!(ESFSignature640::descriptor_size(), 640);
    }

    #[test]
    fn test_gasd512_signature() {
        let gasd = GASDSignature512::new();
        assert_eq!(gasd.gasd.len(), 512);
        assert_eq!(GASDSignature512::descriptor_size(), 512);
    }

    #[test]
    fn test_gasd984_signature() {
        let gasd = GASDSignature984::new();
        assert_eq!(gasd.gasd.len(), 984);
        assert_eq!(GASDSignature984::descriptor_size(), 984);
    }
}

#[cfg(test)]
mod test10 {
    use super::*;

    #[test]
    fn test_gasd_signature() {
        let gasd = GASDSignature7992::new();
        assert_eq!(gasd.gasd.len(), 7992);
    }

    #[test]
    fn test_gfpfh_signature() {
        let gfpfh = GFPFHSignature16::new();
        assert_eq!(gfpfh.gfpfh.len(), 16);
    }

    #[test]
    fn test_brisk_signature() {
        let brisk = BRISKSignature512::with_params(1.0, 0.5);
        assert_eq!(brisk.scale, 1.0);
        assert_eq!(brisk.orientation, 0.5);
        assert_eq!(brisk.descriptor.len(), 64);
    }

    #[test]
    fn test_narf36() {
        let narf = Narf36::with_all_params(1.0, 2.0, 3.0, 0.1, 0.2, 0.3);
        assert_eq!(narf.x, 1.0);
        assert_eq!(narf.y, 2.0);
        assert_eq!(narf.z, 3.0);
        assert_eq!(narf.descriptor.len(), 36);
    }

    #[test]
    fn test_border_description() {
        let border = BorderDescription::with_position(10, 20);
        assert_eq!(border.x, 10);
        assert_eq!(border.y, 20);
        assert_eq!(border.traits.len(), 32);
    }
}

#[cfg(test)]
mod test11 {
    use super::*;

    #[test]
    fn test_intensity_gradient() {
        let grad = IntensityGradient::with_values(1.0, 2.0, 3.0);
        assert_eq!(grad.gradient, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_histogram() {
        let hist: Histogram<3> = Histogram {
            histogram: [0.1, 0.2, 0.3],
        };
        assert_eq!(hist.histogram, [0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_point_with_scale() {
        let point = PointWithScale::with_values(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0);
        assert_eq!(point.x, 1.0);
        assert_eq!(point.octave, 7.0);
    }

    #[test]
    fn test_point_surfel() {
        let surfel = PointSurfel::with_values(
            1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 255, 128, 64, 32, 0.5, 0.8, 0.2,
        );
        assert_eq!(surfel.x, 1.0);
        assert_eq!(surfel.rgba, 0xFF804020);
    }

    #[test]
    fn test_point_dem() {
        let dem = PointDEM::with_values(1.0, 2.0, 3.0, 0.5, 0.1, 0.2);
        assert_eq!(dem.x, 1.0);
        assert_eq!(dem.height_variance, 0.2);
    }
}
