#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]

//! 实现KdPoint特性

// 标准库
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::env;
use std::fmt;

// 宏编程
use paste::paste;

// kd树
use kd_tree::{KdPoint, KdTree};
use typenum;

// 内部库
use crate::common::impls;
use crate::common::PointCloud;
use crate::common::{
    Axis, BRISKSignature512, BorderDescription, BorderTraits, Boundary, CPPFSignature,
    ESFSignature640, FPFHSignature33, GASDSignature512, GASDSignature7992, GASDSignature984,
    GFPFHSignature16, GRSDSignature21, Histogram, Intensity, Intensity32u, Intensity8u,
    IntensityGradient, InterestPoint, Label, MomentInvariants, Narf36, Normal,
    NormalBasedSignature12, PFHRGBSignature250, PFHSignature125, PPFRGBSignature, PPFSignature,
    PointDEM, PointNormal, PointSurfel, PointUV, PointWithRange, PointWithScale,
    PointWithViewpoint, PointXY, PointXYZ, PointXYZHSV, PointXYZI, PointXYZINormal, PointXYZL,
    PointXYZLAB, PointXYZLNormal, PointXYZRGB, PointXYZRGBA, PointXYZRGBL, PointXYZRGBNormal,
    PrincipalCurvatures, PrincipalRadiiRSD, ReferenceFrame, ShapeContext1980,
    UniqueShapeContext1960, VFHSignature308, RGB, SHOT1344, SHOT352,
};

/* start 包装点和id */

// 9. 类型: PointXYZRGBA
#[derive(Debug, Clone)]
pub struct PointXYZRGBAWithId {
    pub id: usize,
    pub point: PointXYZRGBA,
}

// 10. 类型: PointXYZRGB
#[derive(Debug, Clone)]
pub struct PointXYZRGBWithId {
    pub id: usize,
    pub point: PointXYZRGB,
}

// 20. 类型: PointXYZRGBNormal
#[derive(Debug, Clone)]
pub struct PointXYZRGBNormalWithId {
    pub id: usize,
    pub point: PointXYZRGBNormal,
}

/* end 包装点和id */

/* start 实现构造函数 */

// 10. 类型: PointXYZRGB
impl PointXYZRGB {
    /// 检查点是否有效
    pub fn is_valid(&self) -> bool {
        // 这里可以根据具体需求定义点的有效性
        // 例如，检查坐标是否为有限值
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// 将点数据向量化
    pub fn vectorize(&self, data: &mut Vec<f32>) {
        data.push(self.x);
        data.push(self.y);
        data.push(self.z);
    }
}

impl PointXYZRGBWithId {
    // 创建一个新的PointXYZRGBWithId实例
    pub fn new(id: usize, point: PointXYZRGB) -> Self {
        PointXYZRGBWithId { id, point }
    }
}
impl Default for PointXYZRGBWithId {
    // 返回一个默认的PointXYZRGBWithId实例
    fn default() -> Self {
        PointXYZRGBWithId {
            id: 0, // 默认id为0
            point: PointXYZRGB {
                x: 0.0, // 默认x坐标为0.0
                y: 0.0, // 默认y坐标为0.0
                z: 0.0, // 默认z坐标为0.0
                rgb: 0,
            },
        }
    }
}

// 20. 类型: PointXYZRGBNormal
impl PointXYZRGBNormal {
    /// 检查点是否有效
    pub fn is_valid(&self) -> bool {
        // 这里可以根据具体需求定义点的有效性
        // 例如，检查坐标是否为有限值
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// 将点数据向量化
    pub fn vectorize(&self, data: &mut Vec<f32>) {
        data.push(self.x);
        data.push(self.y);
        data.push(self.z);
    }
}

impl PointXYZRGBNormalWithId {
    // 创建一个新的PointXYZRGBWithId实例
    pub fn new(id: usize, point: PointXYZRGBNormal) -> Self {
        PointXYZRGBNormalWithId { id, point }
    }
}
impl Default for PointXYZRGBNormalWithId {
    // 返回一个默认的PointXYZRGBWithId实例
    fn default() -> Self {
        PointXYZRGBNormalWithId {
            id: 0, // 默认id为0
            point: PointXYZRGBNormal {
                x: 0.0, // 默认x坐标为0.0
                y: 0.0, // 默认y坐标为0.0
                z: 0.0, // 默认z坐标为0.0
                rgb: 0, // 颜色
                normal: [0.0, 0.0, 0.0],
                curvature: 0.0,
            },
        }
    }
}

/* end 实现构造函数 */

/* start 实现KdPoint trait */

// 10. 类型: PointXYZRGB
impl KdPoint for PointXYZRGBWithId {
    type Scalar = f32;
    // 3维点云
    type Dim = typenum::U3; 
    fn at(&self, k: usize) -> f32 {
        match k {
            0 => self.point.x,
            1 => self.point.y,
            2 => self.point.z,
            _ => panic!("Invalid dimension"),
        }
    }
    
}

// 20. 类型: PointXYZRGBNormal
impl KdPoint for PointXYZRGBNormalWithId {
    type Scalar = f32;
    // 3维点云
    type Dim = typenum::U3; 
    fn at(&self, k: usize) -> f32 {
        match k {
            0 => self.point.x,
            1 => self.point.y,
            2 => self.point.z,
            _ => panic!("Invalid dimension"),
        }
    }
}

/* end 实现KdPoint trait */
