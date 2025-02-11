#![allow(unused_imports)]

//! 外部测试 KdTreeRust<PointXYZRGBNormalWithId> 的构造函数
use std::sync::Arc;
use bye_pcl_rs::kdtree::KdTreeRust;
use bye_pcl_rs::kdtree::PointXYZRGBNormalWithId;
use bye_pcl_rs::common::PointXYZRGBNormal;

#[test]
fn test_new() {
    let kdtree = KdTreeRust::<PointXYZRGBNormalWithId>::new(true);
    assert_eq!(kdtree.get_sorted_results(), true);
    assert_eq!(kdtree.get_epsilon(), 0.0);
    assert_eq!(kdtree.get_min_pts(), 0);
}

// 测试 set_input_cloud 方法
#[test]
fn test_set_input_cloud() {
    let mut kdtree = KdTreeRust::<PointXYZRGBNormalWithId>::new(true);
    let cloud = Arc::new(vec![PointXYZRGBNormal::new(1.0, 2.0, 3.0, 255, [0.0, 0.0, 0.0], 0.0)]);
    kdtree.set_input_cloud(cloud.clone(), None);
    let input_cloud = kdtree.get_input_cloud();
    assert_eq!(input_cloud.len(), 1);
}

// 测试 nearest_k_search 方法
#[test]
fn test_nearest_k_search() {
    let mut kdtree = KdTreeRust::<PointXYZRGBNormalWithId>::new(true);
    let cloud = Arc::new(vec![
        PointXYZRGBNormal::new(1.0, 2.0, 3.0, 255, [0.0, 0.0, 0.0], 0.0),
        PointXYZRGBNormal::new(4.0, 5.0, 6.0, 255, [0.0, 0.0, 0.0], 0.0),
    ]);
    kdtree.set_input_cloud(cloud.clone(), None);
    let query_point = PointXYZRGBNormalWithId::new(0, PointXYZRGBNormal::new(1.1, 2.1, 3.1, 255, [0.0, 0.0, 0.0], 0.0));
    let (indices, distances) = kdtree.nearest_k_search(&query_point, 1);
    assert_eq!(indices.len(), 1);
    assert_eq!(distances.len(), 1);
}

// 测试 radius_search 方法
#[test]
fn test_radius_search() {
    let mut kdtree = KdTreeRust::<PointXYZRGBNormalWithId>::new(true);
    let cloud = Arc::new(vec![
        PointXYZRGBNormal::new(1.0, 2.0, 3.0, 255, [0.0, 0.0, 0.0], 0.0),
        PointXYZRGBNormal::new(4.0, 5.0, 6.0, 255, [0.0, 0.0, 0.0], 0.0),
    ]);
    kdtree.set_input_cloud(cloud.clone(), None);
    let query_point = PointXYZRGBNormalWithId::new(0, PointXYZRGBNormal::new(1.1, 2.1, 3.1, 255, [0.0, 0.0, 0.0], 0.0));
    let (indices, distances) = kdtree.radius_search(&query_point, 1.0, 10);
    assert_eq!(indices.len(), 1);
    assert_eq!(distances.len(), 1);
}

// 测试 set_epsilon 和 get_epsilon 方法
#[test]
fn test_epsilon() {
    let mut kdtree = KdTreeRust::<PointXYZRGBNormalWithId>::new(true);
    kdtree.set_epsilon(0.1);
    assert_eq!(kdtree.get_epsilon(), 0.1);
}

// 测试 set_min_pts 和 get_min_pts 方法
#[test]
fn test_min_pts() {
    let mut kdtree = KdTreeRust::<PointXYZRGBNormalWithId>::new(true);
    kdtree.set_min_pts(5);
    assert_eq!(kdtree.get_min_pts(), 5);
}
