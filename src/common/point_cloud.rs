#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]

//! 点云结构体

// 标准库
use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

// 元编程(代码生成)
use paste::paste;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

// 线性代数
use nalgebra::{DMatrix, DVector, Point3, Quaternion, Vector4};

// 导入内部库
// #[path = "./pcl_header.rs"]
use super::pcl_header::PclHeader;

// 1. 属性映射结构体
#[derive(Default)]
struct FieldMapping {
    serialized_offset: usize,
    struct_offset: usize,
    size: usize,
}
type MsgFieldMap = Vec<FieldMapping>;

// 2. 适配点云结构体和nalgebra类型的中间层
struct NdCopyEigenPointFunctor<PointOutputType> {
    // _m: PhantomData<PointOutputType>,
    p1: DVector<f32>,
    p2: *mut PointOutputType,
    f_idx: usize,
}

impl<PointOutputType> NdCopyEigenPointFunctor<PointOutputType> {
    fn new(p1: DVector<f32>, p2: &mut PointOutputType) -> Self {
        Self {
            p1,
            p2: p2 as *mut PointOutputType,
            f_idx: 0,
        }
    }
}

// 3. 用于在 PointT 和 nalgebra 类型之间复制数据的辅助结构
struct NdCopyPointEigenFunctor<PointInT> {
    p1: *const PointInT,
    p2: DVector<f32>,
    f_idx: usize,
}

impl<PointInT> NdCopyPointEigenFunctor<PointInT> {
    fn new(p1: &PointInT, p2: DVector<f32>) -> Self {
        Self {
            p1: p1 as *const PointInT,
            p2,
            f_idx: 0,
        }
    }
}

// 4. 点云结构体
#[derive(Clone, Debug, Default)]
pub struct PointCloud<PointType> {
    // 点云头信息，包含采集时间等信息
    pub header: PclHeader,
    /// 点数据
    pub points: Vec<PointType>,
    /// 点云数量
    pub size: u32,
    /// 点云宽度（如果以图像结构组织）
    pub width: u32,
    /// 点云高度（如果以图像结构组织）
    pub height: u32,
    /// 是否所有点都是有效的（没有NaN或Inf值）
    pub is_dense: bool,
    /// 传感器采集位置（原点/平移）
    pub sensor_origin: Vector4<f32>,
    /// 传感器采集姿态（旋转）
    pub sensor_orientation: Quaternion<f32>,
    /// 占位符
    _m: PhantomData<PointType>,
}

impl<PointType: Clone> PointCloud<PointType> {
    // 构造函数
    pub fn new() -> Self {
        PointCloud {
            header: PclHeader::new_auto(),
            points: Vec::<PointType>::new(),
            size: 0,
            width: 0,
            height: 0,
            is_dense: true,
            sensor_origin: Vector4::zeros(),
            sensor_orientation: Quaternion::identity(),
            _m: PhantomData::<PointType>,
        }
    }
    /// 从点云容器构造
    pub fn from_points_vec(input_points_vec: Vec<PointType>) -> Self {
        let len = input_points_vec.len() as u32;
        Self {
            header: PclHeader::new_auto(),
            points: input_points_vec,
            size: len,
            width: len,
            height: 1,
            is_dense: true,
            sensor_origin: Vector4::zeros(),
            sensor_orientation: Quaternion::identity(),
            _m: PhantomData::<PointType>,
        }
    }

    /// 从点云子集构造
    pub fn from_subset(pc: &PointCloud<PointType>, indices: &[usize]) -> Self {
        let mut points = Vec::with_capacity(indices.len());
        for &i in indices {
            points.push(pc.points[i].clone());
        }
        let len = indices.len() as u32;
        PointCloud {
            header: pc.header.clone(),
            points,
            size: len,
            width: len,
            height: 1,
            is_dense: pc.is_dense,
            sensor_origin: pc.sensor_origin,
            sensor_orientation: pc.sensor_orientation,
            _m: PhantomData::<PointType>,
        }
    }

    /// 获取点云数量
    pub fn size(&self) -> usize {
        self.points.len()
    }

    /// 判断点云是否是有组织的
    pub fn is_organized(&self) -> bool {
        self.height > 1
    }

    /// 清空点云
    pub fn clear(&mut self) {
        self.points.clear();
        self.width = 0;
        self.height = 0;
    }
}

// 点云输出格式化
impl<PointType> fmt::Display for PointCloud<PointType> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "header: {}", self.header)?;
        writeln!(f, "points[]: {}", self.size)?;
        writeln!(f, "width: {}", self.width)?;
        writeln!(f, "height: {}", self.height)?;
        writeln!(f, "is_dense: {}", self.is_dense)?;
        writeln!(
            f,
            "sensor origin (xyz): [{}, {}, {}] / orientation (xyzw): [{}, {}, {}, {}]",
            self.sensor_origin.x,
            self.sensor_origin.y,
            self.sensor_origin.z,
            self.sensor_orientation.i,
            self.sensor_orientation.j,
            self.sensor_orientation.k,
            self.sensor_orientation.w
        )
    }
}
