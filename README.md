# 🌟bye_pcl_rs点云库🌟/Point Cloud Library

**点云库仍然在开发中, 功能实现不完全/Unimplemented!!!**
**提示: 由于功能实现不完全，因而临时导入了f3l库**
**Notice: use f3l crate**

## 📝简介/Description
点云库专门用于2D/3D图像和点云处理。🚀 使用纯Rust ⚙️ 进行编写。
The Point Cloud Library (PCL) is a standalone, large scale, open project for 2D/3D image and point cloud processing.
Implemented in pure Rust.

👉快来体验PCL的强大功能，为你的研究添砖加瓦吧！💪💡
#点云库 #PCL #3D图像处理 #研究工具 #开源项目

## 📖参考/Citation
```bibtex
@InProceedings{Rusu_ICRA2011_PCL,
  author    = {Radu Bogdan Rusu and Steve Cousins},
  title     = {{3D is here: Point Cloud Library (PCL)}},
  booktitle = {{IEEE International Conference on Robotics and Automation (ICRA)}},
  month     = {May 9-13},
  year      = {2011},
  address   = {Shanghai, China},
  publisher = {IEEE}
}
```

## 🎉示例/Example
- bye_pcl_rs version 0.1.1
```sh
cargo run --example surfel_mapping
```
代码:
```rs
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(non_fmt_panics)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![allow(rustdoc::missing_crate_level_docs)]
#![allow(unsafe_code)]
#![allow(clippy::undocumented_unsafe_blocks)]
#![allow(unused_must_use)]
#![allow(non_snake_case)]
#![allow(unused_doc_comments)]

//! 从点云数据中重建三维表面并生成网格:
//! 1. **加载点云数据**：程序首先从PCD文件中加载点云数据，这些数据包含了三维空间中的点及其颜色信息。
//! 2. **表面重建**：使用移动最小二乘法（MLS）对点云进行平滑处理，并计算每个点的法向量。这一步生成了带有法向量的点云数据，称为“表面元素”（surfels）。
//! 3. **网格生成**：通过贪婪投影三角化算法（Greedy Projection Triangulation）将表面元素转换为三角网格。这个算法会根据点的位置和法向量生成三角形，从而构建出三维表面。
//! 4. **可视化**：最后，程序使用PCL的可视化工具将生成的网格显示出来，用户可以通过交互方式查看三维模型。

use std::fs::create_dir_all;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Cursor, Write};
use std::path::Path;
use std::sync::Arc;

use std::cell::{Ref, RefCell, RefMut};

// 线性代数
use nalgebra::{
    Const, DMatrix, DVector, Isometry3, Matrix, Matrix2, Matrix3, Matrix3x1, Matrix6, Point3,
    Quaternion, Rotation3, SVector, Translation3, UnitQuaternion, Vector2, Vector3, Vector6,
    VectorView3, ViewStorage,
};

// 随机数
use rand::Rng;
use rand_distr::{Distribution, Normal};

// 点云处理
use bye_pcl_rs::{
    common::{PointXYZRGB, PointCloud, PointXYZRGBNormal},
    io::pcd_io,
    kdtree::{KdTreeRust, PointXYZRGBWithId, PointXYZRGBNormalWithId},
    surface::{MovingLeastSquares, PolygonMesh, GreedyProjectionTriangulation},
};

// 1. 重建表面, 输入的PointXYZRGBNormal的normal为空, 其实就是PointXYZRGB
fn reconstruct_surface(input: &PointCloud<PointXYZRGB>, radius: f64, polynomial_order: usize) -> PointCloud<PointXYZRGBNormal> {
    // 创建 MovingLeastSquares 实例，用于最小二乘表面重建
    let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();

    // 创建 KdTreeRust 实例，用于搜索最近邻
    let mut kdtree = KdTreeRust::<PointXYZRGBWithId>::new(true);

    // 将输入点云设置到 KdTreeRust 中
    let input_arc = input.points.clone().into();
    kdtree.set_input_cloud(input_arc, None);

    // 这里由于没有直接的 setSearchMethod 方法，我们可以考虑在 MovingLeastSquares 内部使用 KdTreeRust 进行搜索
    // 但为了简化，我们在 MovingLeastSquares 的 find_neighbors 方法中使用 KdTreeRust 来查找邻居

    // 设置搜索半径，用于确定每个点的邻域范围
    mls.set_search_radius(radius);

    // 设置是否计算法线，这里设置为 true 表示计算法线
    mls.set_compute_normals(true);

    // 设置多项式阶数，用于多项式拟合
    mls.set_polynomial_order(polynomial_order);

    // 设置输入点云，将待处理的点云传递给 MovingLeastSquares 实例
    let new_points: Vec<PointXYZRGBNormal> = input.points.clone().into_iter().map(|p| {
        let mut new_point = PointXYZRGBNormal::default();
        new_point.x = p.x;
        new_point.y = p.y;
        new_point.z = p.z;
        new_point.rgb = p.rgb;
        // 返回值
        new_point
    }).collect();
    let arc_points = Arc::new(new_points);
    mls.set_input_cloud(arc_points);

    // 创建输出点云实例，用于存储重建后的表面点云
    let mut output = PointCloud::<PointXYZRGBNormal>::default();

    // 调用 process 方法进行表面重建处理
    mls.process();

    // 将处理后的结果赋值给输出点云
    output=PointCloud::<PointXYZRGBNormal>::from_points_vec(mls.output.clone());

    // 返回重建后的表面点云
    output
}

// 2. 网格三角化
fn triangulate_mesh(surfels: &PointCloud<PointXYZRGBNormalWithId>) -> PolygonMesh {
    // 创建 KdTreeRust 实例，用于搜索最近邻
    let mut kdtree = KdTreeRust::<PointXYZRGBNormalWithId>::new(true);
    
    // 将 Vec<PointXYZRGBNormalWithId> 转换为 Vec<PointXYZRGBNormal>
    let points: Vec<PointXYZRGBNormal> = surfels.points.iter().map(|p| p.point.clone()).collect();
    // 将 Vec<PointXYZRGBNormal> 转换为 Arc<Vec<PointXYZRGBNormal>>
    let surfels_arc = Arc::new(points);

    // 将输入点云设置到 KdTreeRust 中
    kdtree.set_input_cloud(surfels_arc, None);

    // 初始化贪婪投影三角化对象
    let mut gp3 = GreedyProjectionTriangulation::<PointXYZRGBNormalWithId>::new();
    // 初始化多边形网格对象，用于存储三角化结果
    let mut triangles = PolygonMesh::default();

    // 设置连接点之间的最大距离（最大边长）
    gp3.set_search_radius(0.05);

    // 设置参数的典型值
    // 设置最近邻距离乘数
    gp3.set_mu(2.5);
    // 设置最大最近邻数量
    gp3.set_maximum_nearest_neighbors(100);
    // 设置最大表面角度（45度）
    gp3.set_maximum_surface_angle(std::f64::consts::PI / 4.0);
    // 设置三角形的最小角度（10度）
    gp3.set_minimum_angle(std::f64::consts::PI / 18.0);
    // 设置三角形的最大角度（120度）
    gp3.set_maximum_angle(2.0 * std::f64::consts::PI / 3.0);
    // 设置输入法线是否一致定向的标志
    gp3.set_normal_consistency(true);

    // 设置输入点云
    gp3.input = Some(surfels.clone());
    // 设置搜索方法
    gp3.kdtree = kdtree;

    // 获取三角化结果
    gp3.perform_reconstruction(&mut triangles);

    // 返回三角化后的多边形网格
    triangles
}

// 3. 主函数
fn main() {
    // 1. 加载ply点云文件(pcd文件解析有问题)到点云结构体
    let path = "./assets/office1.ply";
    let result = PointCloud::<PointXYZRGB>::load_from_ply(path);
    // 断言加载成功
    assert!(result.is_ok(), "加载PLY文件失败: {:?}", result.err());
    let point_cloud = result.unwrap();
    // 验证点云数据不为空
    assert!(!point_cloud.points.is_empty(), "点云数据为空");
    
    // 2. 获取点云数量
    println!("成功加载 {} 个点", point_cloud.points.len());

    // 3. 计算点云面元素
    println!("正在计算法线 ... ");
    let mls_radius = 0.05;
    let polynomial_order = 2;
    let point_cloud_with_normal = reconstruct_surface(&point_cloud, mls_radius, polynomial_order);
    // 将 PointXYZRGBNormal 转换为 PointXYZRGBNormalWithId
    let mut surfels = PointCloud::<PointXYZRGBNormalWithId>::default();
    surfels.points = point_cloud_with_normal.points.iter().enumerate().map(|(id, point)| {
        PointXYZRGBNormalWithId {
            id,
            point: point.clone(),
        }
    }).collect();
    surfels.is_dense = point_cloud_with_normal.is_dense;

    // 4. 贪婪三角化
    println!("正在计算网格 ... ");
    let mesh = triangulate_mesh(&surfels);
    
    // 5. 输出结果
    println!("三角化完成，网格中有 {} 个多边形", mesh.polygons.len());

    // 6. 显示点云网格
    println!("显示点云网格使用其他库, 例如bevy, pasture\n");
    println!("unimplemented!()");
}
```
