#![allow(unused_imports)]

use bye_pcl_rs::common::{
    PointCloud, PointXYZRGBNormal, 
};

use bye_pcl_rs::kdtree::{
    PointXYZRGBNormalWithId, PointXYZRGBWithId,
};

use bye_pcl_rs::surface::GreedyProjectionTriangulation;

// 辅助函数：创建一个简单的点云
fn create_simple_point_cloud() -> PointCloud<PointXYZRGBNormalWithId> {
    let mut point_cloud = PointCloud::<PointXYZRGBNormalWithId>::default();
    // 添加一些示例点
    for i in 0..10 {
        let point = PointXYZRGBNormalWithId {
            point: PointXYZRGBNormal {
                x: i as f32,
                y: 0.0,
                z: 0.0,
                // 使用 rgb 字段，这里简单将其设为 0
                rgb: 0,
                curvature: 0.0,
                // 使用 normal 字段，类型为 [f32; 3]
                normal: [0.0, 1.0, 0.0],
            },
            id: i,
        };
        point_cloud.points.push(point);
    }
    point_cloud.is_dense = true;
    point_cloud
}

#[test]
// 测试 GreedyProjectionTriangulation 的空构造函数
fn test_new() {
    let gp3 = GreedyProjectionTriangulation::<PointXYZRGBNormalWithId>::new();
    assert_eq!(gp3.mu, 0.0);
    assert_eq!(gp3.search_radius, 0.0);
    assert_eq!(gp3.nnn, 100);
    assert_eq!(gp3.minimum_angle, std::f64::consts::PI / 18.0);
    assert_eq!(gp3.maximum_angle, 2.0 * std::f64::consts::PI / 3.0);
    assert_eq!(gp3.eps_angle, std::f64::consts::PI / 4.0);
    assert!(!gp3.consistent);
    assert!(!gp3.consistent_ordering);
}

#[test]
// 测试设置和获取最近邻距离乘数
fn test_set_and_get_mu() {
    let mut gp3 = GreedyProjectionTriangulation::<PointXYZRGBNormalWithId>::new();
    let new_mu = 1.5;
    gp3.set_mu(new_mu);
    assert_eq!(gp3.get_mu(), new_mu);
}

#[test]
// 测试设置和获取最大最近邻数量
fn test_set_and_get_maximum_nearest_neighbors() {
    let mut gp3 = GreedyProjectionTriangulation::<PointXYZRGBNormalWithId>::new();
    let new_nnn = 50;
    gp3.set_maximum_nearest_neighbors(new_nnn);
    assert_eq!(gp3.get_maximum_nearest_neighbors(), new_nnn);
}

#[test]
// 测试设置和获取搜索半径
fn test_set_and_get_search_radius() {
    let mut gp3 = GreedyProjectionTriangulation::<PointXYZRGBNormalWithId>::new();
    let new_search_radius = 2.0;
    gp3.set_search_radius(new_search_radius);
    assert_eq!(gp3.get_search_radius(), new_search_radius);
}

#[test]
// 测试设置和获取三角形的最小角度
fn test_set_and_get_minimum_angle() {
    let mut gp3 = GreedyProjectionTriangulation::<PointXYZRGBNormalWithId>::new();
    let new_minimum_angle = std::f64::consts::PI / 9.0;
    gp3.set_minimum_angle(new_minimum_angle);
    assert_eq!(gp3.get_minimum_angle(), new_minimum_angle);
}

#[test]
// 测试设置和获取三角形的最大角度
fn test_set_and_get_maximum_angle() {
    let mut gp3 = GreedyProjectionTriangulation::<PointXYZRGBNormalWithId>::new();
    let new_maximum_angle = std::f64::consts::PI;
    gp3.set_maximum_angle(new_maximum_angle);
    assert_eq!(gp3.get_maximum_angle(), new_maximum_angle);
}

#[test]
// 测试设置和获取最大表面角度
fn test_set_and_get_maximum_surface_angle() {
    let mut gp3 = GreedyProjectionTriangulation::<PointXYZRGBNormalWithId>::new();
    let new_eps_angle = std::f64::consts::PI / 3.0;
    gp3.set_maximum_surface_angle(new_eps_angle);
    assert_eq!(gp3.get_maximum_surface_angle(), new_eps_angle);
}

#[test]
// 测试设置和获取输入法线是否一致定向的标志
fn test_set_and_get_normal_consistency() {
    let mut gp3 = GreedyProjectionTriangulation::<PointXYZRGBNormalWithId>::new();
    let new_consistent = true;
    gp3.set_normal_consistency(new_consistent);
    assert_eq!(gp3.get_normal_consistency(), new_consistent);
}

#[test]
// 测试设置和获取输出三角形顶点是否应一致定向的标志
fn test_set_and_get_consistent_vertex_ordering() {
    let mut gp3 = GreedyProjectionTriangulation::<PointXYZRGBNormalWithId>::new();
    let new_consistent_ordering = true;
    gp3.set_consistent_vertex_ordering(new_consistent_ordering);
    assert_eq!(gp3.get_consistent_vertex_ordering(), new_consistent_ordering);
}

#[test]
// 测试执行表面重建，输出多边形列表
fn test_perform_reconstruction_polygons() {
    let mut gp3 = GreedyProjectionTriangulation::<PointXYZRGBNormalWithId>::new();
    let point_cloud = create_simple_point_cloud();
    gp3.input = Some(point_cloud);
    gp3.indices = Some((0..gp3.input.as_ref().unwrap().points.len()).collect());
    gp3.set_search_radius(1.0);
    gp3.set_mu(1.0);
    let mut polygons = Vec::new();
    gp3.perform_reconstruction_polygons(&mut polygons);
    // 这里简单检查是否有生成的多边形
    assert!(!polygons.is_empty());
}
