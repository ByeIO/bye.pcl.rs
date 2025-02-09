#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]

//! 加载pcd文件

// 标准库
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::env;
use std::fmt;

// 宏编程
use paste::paste;
// 点云PCD文件处理
use bye_pcd_rs::{PcdDeserialize, Reader};
use eyre::Result;
// 点云PLY文件处理
extern crate ply_rs_bw;
use ply_rs_bw as ply;
use ply_rs_bw::ply::Property;

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

/* start PCD文件 */
pub struct PcdMeta {
    pub version: String,
    pub width: u64,
    pub height: u64,
    pub viewpoint: ViewPoint,
    pub num_points: u64,
    pub data: PcdDataKind,
    pub field_defs: bye_pcd_rs::Schema,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ViewPoint {
    pub tx: f64,
    pub ty: f64,
    pub tz: f64,
    pub qw: f64,
    pub qx: f64,
    pub qy: f64,
    pub qz: f64,
}

impl Default for ViewPoint {
    fn default() -> Self {
        ViewPoint {
            tx: 0.0,
            ty: 0.0,
            tz: 0.0,
            qw: 1.0,
            qx: 0.0,
            qy: 0.0,
            qz: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcdDataKind {
    Ascii,
    Binary,
}
/* end PCD文件 */

/* start 点云导入 */

// 1. 为点云类型实现PointCloud从pcd加载功能, 
// 使用paste为所有点云类型生成impl块
macro_rules! impl_point_cloud_pcd {
    ($($type:ty),*) => {
        $(
            paste! {
                impl PointCloud<$type> {
                    // 加载点云入口函数
                    pub fn load_from_pcd(path: &str) -> Result<Self, std::io::Error> {
                        let path = Path::new(path);
                        // 打印工作目录
                        let current_dir = env::current_dir().expect("Failed to get current directory");
                        println!("工作目录: {}\n", current_dir.display());
                        // 打印获取到绝对路径
                        let absolute_path = std::fs::canonicalize(&path).expect("Failed to get absolute path");
                        println!("PCD文件: {}\n", absolute_path.display());
                             
                        // binary_compressed的格式
                        let reader = bye_pcd_rs::Reader::open(path)
                            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                        let collected_points: eyre::Result<Vec<$type>, _> = reader.collect();
                        // 从点云容器构造点云并返回
                        Ok(PointCloud::from_points_vec(collected_points.unwrap()))
                    }
                }
            }
        )*
    };
}

// 为所有给定的点云类型生成impl块
impl_point_cloud_pcd!(
    PointDEM, PointNormal, PointSurfel, PointUV, PointWithRange, PointWithScale,
    PointWithViewpoint, PointXY, PointXYZ, PointXYZHSV, PointXYZI, PointXYZINormal, PointXYZL,
    PointXYZLAB, PointXYZLNormal, PointXYZRGB, PointXYZRGBA, PointXYZRGBL, PointXYZRGBNormal
);

// 2. 为点云类型实现PointCloud从ply加载功能, 
macro_rules! impl_point_cloud_ply {
    ($($type:ty),*) => {
        $(
            paste! {
                impl PointCloud<$type> {
                    // 加载点云入口函数
                    pub fn load_from_ply(path: &str) -> Result<Self, std::io::Error> {
                        let path = Path::new(path);
                        // 打印工作目录
                        let current_dir = env::current_dir().expect("Failed to get current directory");
                        println!("工作目录: {}\n", current_dir.display());
                        // 打印获取到绝对路径
                        let absolute_path = std::fs::canonicalize(&path).expect("Failed to get absolute path");
                        println!("PLY文件: {}\n", absolute_path.display());

                        // 打开PLY文件
                        let mut f = std::fs::File::open(path).unwrap();
                        // 创建PLY解析器
                        let p = ply::parser::Parser::<ply::ply::DefaultElement>::new();
                        // 读取PLY文件
                        let ply = p.read_ply(&mut f);

                        // 检查是否读取成功
                        if ply.is_err() {
                            return Err(std::io::Error::new(std::io::ErrorKind::Other, "Failed to read PLY file"));
                        }

                        let ply = ply.unwrap();
                        // 将PLY数据转换为点云数据
                        let points: Vec<$type> = ply.payload.into_iter()
                            .flat_map(|element| {
                                let mut valid_points = Vec::new();
                                match stringify!($type) {
                                    "PointXYZRGB" => {
                                        if !element.1.is_empty() {
                                            let mut i: usize = 0;
                                            for map in &element.1 {
                                                // println!("循环到第{i}个元素");
                                                i += 1;
                        
                                                let mut x = f32::NAN;
                                                let mut y = f32::NAN;
                                                let mut z = f32::NAN;
                                                let mut red = 0u8;
                                                let mut green = 0u8;
                                                let mut blue = 0u8;
                        
                                                for (key, value) in map {
                                                    // println!("{}: {:?}", key, value);
                                                    match key.as_str() {
                                                        "x" | "y" | "z" => {
                                                            if let Property::Float(f) = value {
                                                                if !f.is_nan() {
                                                                    match key.as_str() {
                                                                        "x" => x = *f,
                                                                        "y" => y = *f,
                                                                        "z" => z = *f,
                                                                        _ => {}
                                                                    }
                                                                }
                                                            }
                                                        },
                                                        "red" => {
                                                            if let Property::UChar(u) = value {
                                                                red = *u;
                                                            }
                                                        },
                                                        "green" => {
                                                            if let Property::UChar(u) = value {
                                                                green = *u;
                                                            }
                                                        },
                                                        "blue" => {
                                                            if let Property::UChar(u) = value {
                                                                blue = *u;
                                                            }
                                                        },
                                                        _ => {}
                                                    }
                                                }
                        
                                                if !x.is_nan() && !y.is_nan() && !z.is_nan() {
                                                    let rgb = ((red as u32) << 16) | ((green as u32) << 8) | (blue as u32);
                                                    valid_points.push($type::new(x, y, z, rgb));
                                                }
                                            }
                                        }
                                    },
                                    _ => panic!("不支持的点类型: {}", stringify!($type)),
                                }
                                valid_points
                            })
                            .collect();

                        // 从点云容器构造点云并返回
                        Ok(PointCloud::from_points_vec(points))
                    }
                }
            }
        )*
    };
}

// 为所有给定的点云类型生成impl块
impl_point_cloud_ply!(
    PointXYZRGB
    // PointDEM, PointNormal, PointSurfel, PointUV, PointWithRange, PointWithScale,
    // PointWithViewpoint, PointXY, PointXYZ, PointXYZHSV, PointXYZI, PointXYZINormal, PointXYZL,
    // PointXYZLAB, PointXYZLNormal, PointXYZRGBA, PointXYZRGBL, PointXYZRGBNormal
);

// /* end 点云导入 */

// /* start 点云导出 */
// // PointCloud保存到pcd文件
// // pub fn save_to_pcd();

// /* end 点云导出 */

#[cfg(test)]
mod tests1 {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_load_pcd_binary_compressed() {
        // 测试加载二进制压缩格式的PCD文件
        let path = "./assets/office1.pcd";
        let result = PointCloud::<PointXYZRGB>::load_from_pcd(path);
        
        // 断言加载成功(pcd-rs不支持binary_compressed), 这里会panic
        assert!(result.is_ok(), "加载PCD文件失败: {:?}", result.err());
        
        let point_cloud = result.unwrap();
        
        // 验证点云数据不为空
        assert!(!point_cloud.points.is_empty(), "点云数据为空");
        
        // 打印加载的点云数量
        println!("成功加载 {} 个点", point_cloud.points.len());
    }
    
    #[test]
    fn test_load_ply() {
        // 测试加载二进制压缩格式的PCD文件
        let path = "./assets/office1.ply";
        let result = PointCloud::<PointXYZRGB>::load_from_ply(path);
        
        // 断言加载成功
        assert!(result.is_ok(), "加载PLY文件失败: {:?}", result.err());
        
        let point_cloud = result.unwrap();
        
        // 验证点云数据不为空
        assert!(!point_cloud.points.is_empty(), "点云数据为空");
        
        // 打印加载的点云数量
        println!("成功加载 {} 个点", point_cloud.points.len());
        
        // println!("点云: {:#?}\n",point_cloud.points);
    }

}
