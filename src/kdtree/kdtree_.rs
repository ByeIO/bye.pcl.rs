#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]

//! 适配kd_tree库的功能, 提供与cpp代码类似的接口

// kd树
use kd_tree::{KdPoint, KdTree};
use typenum;

// 标准库
use std::sync::Arc;

// 内部库
use super::{
  PointXYZRGBWithId,PointXYZRGBAWithId,  PointXYZRGBNormalWithId
};
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

/// KdTree 表示 kd-tree 实现的基础空间定位器类
#[derive(Default, Debug, Clone)]
pub struct KdTreeRust<PointT: KdPoint> {
    // 输入点云数据集
    input: Arc<Vec<PointT>>, 
    // 点索引子集
    indices: Option<Arc<Vec<usize>>>, 
    // 最近邻搜索的精度（误差界限）
    epsilon: f32, 
    // 可行结果必须包含的最小邻居数
    min_pts: usize, 
    // 是否对结果进行排序
    sorted: bool, 
}

/* start 对每一种点都实现kdTreeRust */

// 10. 类型: PointXYZRGB
impl KdTreeRust<PointXYZRGBWithId> {
    
    /// 空构造函数，设置一些内部值为默认值
    pub fn new(sorted: bool) -> Self {
        Self {
            input: Arc::new(Vec::new()),
            // 初始化 indices 字段
            epsilon: 0.0,
            indices: None, 
            min_pts: 0,
            sorted,
        }
    }

    /// 提供输入数据集的指针
    pub fn set_input_cloud(&mut self, cloud: Arc<Vec<PointXYZRGB>>, indices: Option<Arc<Vec<usize>>>) {
        let points_with_id: Vec<PointXYZRGBWithId> = cloud.iter().map(|p|
            PointXYZRGBWithId::new(0,p.clone())).collect(); 
        self.input = Arc::new(points_with_id);
        self.indices = indices;
    }

    /// 获取使用的索引向量的指针
    pub fn get_indices(&self) -> Option<Arc<Vec<usize>>> {
        self.indices.clone()
    }

    /// 获取输入点云数据集的指针
    pub fn get_input_cloud(&self) -> Arc<Vec<PointXYZRGBWithId>> {
        self.input.clone()
    }

    /// 设置最近邻搜索的精度(误差界限)
    pub fn set_epsilon(&mut self, eps: f32) {
        self.epsilon = eps;
    }

    /// 获取最近邻搜索的精度（误差界限）
    pub fn get_epsilon(&self) -> f32 {
        self.epsilon
    }

    /// 设置可行结果必须包含的最小邻居数
    pub fn set_min_pts(&mut self, min_pts: usize) {
        self.min_pts = min_pts;
    }

    /// 获取可行结果必须包含的最小邻居数
    pub fn get_min_pts(&self) -> usize {
        self.min_pts
    }

    /// 获取结果是否应排序（按距离升序）
    pub fn get_sorted_results(&self) -> bool {
        self.sorted
    }

    /// 搜索给定查询点的 k 个最近邻
    pub fn nearest_k_search(&self, point: &PointXYZRGBWithId, k: usize) -> (Vec<usize>, Vec<f32>) {
        let kdtree = KdTree::build_by_ordered_float(self.input.iter().cloned().collect());
        let nearest = kdtree.nearest(point).unwrap();
        // 直接访问 item 字段
        let indices = vec![nearest.item.id]; 
        // 直接访问 squared_distance 字段
        let distances = vec![nearest.squared_distance]; 
        (indices, distances)
    }
    
    /// 搜索给定查询点的 k 个最近邻
    pub fn knn_search(&self, point: &PointXYZRGBWithId, k: usize) -> (Vec<usize>, Vec<f32>) {
        let kdtree = KdTree::build_by_ordered_float(self.input.iter().cloned().collect());
        let nearest = kdtree.nearest(point).unwrap();
        let indices = vec![nearest.item.id];
        let distances = vec![nearest.squared_distance];
        (indices, distances)
    }

    /// 在给定半径内搜索查询点的所有最近邻
    pub fn radius_search(&self, point: &PointXYZRGBWithId, radius: f64, max_nn: usize) -> (Vec<usize>, Vec<f32>) {
        // 构建 KD 树
        let kdtree = KdTree::build_by_ordered_float(self.input.iter().cloned().collect());
        
        // 在半径内搜索最近邻
        let nearest = kdtree.within_radius(point, radius as f32);
        
        // 提取最近邻的 id
        let indices = nearest.iter().map(|n| n.id).collect();
        
        // 计算最近邻与查询点之间的欧几里得距离
        let distances = nearest.iter().map(|n| {
            let dx = n.point.x - point.point.x;
            let dy = n.point.y - point.point.y;
            let dz = n.point.z - point.point.z;
            // 计算欧几里得距离
            (dx * dx + dy * dy + dz * dz).sqrt() 
        }).collect();
        
        // 返回最近邻的索引和距离
        (indices, distances)
    }
    
}

// 20. 类型: PointXYZRGBNormal
impl KdTreeRust<PointXYZRGBNormalWithId> {
    
    /// 空构造函数，设置一些内部值为默认值
    pub fn new(sorted: bool) -> Self {
        Self {
            input: Arc::new(Vec::new()),
            // 初始化 indices 字段
            epsilon: 0.0,
            indices: None, 
            min_pts: 0,
            sorted,
        }
    }

    /// 提供输入数据集的指针
    pub fn set_input_cloud(&mut self, cloud: Arc<Vec<PointXYZRGBNormal>>, indices: Option<Arc<Vec<usize>>>) {
        let points_with_id: Vec<PointXYZRGBNormalWithId> = cloud.iter().map(|p|
            PointXYZRGBNormalWithId::new(0,p.clone())).collect(); 
        self.input = Arc::new(points_with_id);
        self.indices = indices;
    }

    /// 获取使用的索引向量的指针
    pub fn get_indices(&self) -> Option<Arc<Vec<usize>>> {
        self.indices.clone()
    }

    /// 获取输入点云数据集的指针
    pub fn get_input_cloud(&self) -> Arc<Vec<PointXYZRGBNormalWithId>> {
        self.input.clone()
    }

    /// 设置最近邻搜索的精度(误差界限)
    pub fn set_epsilon(&mut self, eps: f32) {
        self.epsilon = eps;
    }

    /// 获取最近邻搜索的精度（误差界限）
    pub fn get_epsilon(&self) -> f32 {
        self.epsilon
    }

    /// 设置可行结果必须包含的最小邻居数
    pub fn set_min_pts(&mut self, min_pts: usize) {
        self.min_pts = min_pts;
    }

    /// 获取可行结果必须包含的最小邻居数
    pub fn get_min_pts(&self) -> usize {
        self.min_pts
    }

    /// 获取结果是否应排序（按距离升序）
    pub fn get_sorted_results(&self) -> bool {
        self.sorted
    }

    /// 搜索给定查询点的 k 个最近邻
    pub fn nearest_k_search(&self, point: &PointXYZRGBNormalWithId, k: usize) -> (Vec<usize>, Vec<f32>) {
        let kdtree = KdTree::build_by_ordered_float(self.input.iter().cloned().collect());
        let nearest = kdtree.nearest(point).unwrap();
        // 直接访问 item 字段
        let indices = vec![nearest.item.id]; 
        // 直接访问 squared_distance 字段
        let distances = vec![nearest.squared_distance]; 
        (indices, distances)
    }

    /// 在给定半径内搜索查询点的所有最近邻
    pub fn radius_search(&self, point: &PointXYZRGBNormalWithId, radius: f64, max_nn: usize) -> (Vec<usize>, Vec<f32>) {
        // 构建 KD 树
        let kdtree = KdTree::build_by_ordered_float(self.input.iter().cloned().collect());
        
        // 在半径内搜索最近邻
        let nearest = kdtree.within_radius(point, radius as f32);
        
        // 提取最近邻的 id
        let indices = nearest.iter().map(|n| n.id).collect();
        
        // 计算最近邻与查询点之间的欧几里得距离
        let distances = nearest.iter().map(|n| {
            let dx = n.point.x - point.point.x;
            let dy = n.point.y - point.point.y;
            let dz = n.point.z - point.point.z;
            // 计算欧几里得距离
            (dx * dx + dy * dy + dz * dz).sqrt() 
        }).collect();
        
        // 返回最近邻的索引和距离
        (indices, distances)
    }
    
}

/* end 对每一种点都实现kdTreeRust */

#[cfg(test)]
mod tests10 {
    use super::*;
    use std::sync::Arc;

    // 测试 KdTreeRust<PointXYZRGBWithId> 的构造函数
    #[test]
    fn test_new() {
        let kdtree = KdTreeRust::<PointXYZRGBWithId>::new(true);
        assert_eq!(kdtree.get_sorted_results(), true);
        assert_eq!(kdtree.get_epsilon(), 0.0);
        assert_eq!(kdtree.get_min_pts(), 0);
    }

    // 测试 set_input_cloud 方法
    #[test]
    fn test_set_input_cloud() {
        let mut kdtree = KdTreeRust::<PointXYZRGBWithId>::new(true);
        let cloud = Arc::new(vec![PointXYZRGB::new(1.0, 2.0, 3.0, 255)]);
        kdtree.set_input_cloud(cloud.clone(), None);
        let input_cloud = kdtree.get_input_cloud();
        assert_eq!(input_cloud.len(), 1);
    }

    // 测试 nearest_k_search 方法
    #[test]
    fn test_nearest_k_search() {
        let mut kdtree = KdTreeRust::<PointXYZRGBWithId>::new(true);
        let cloud = Arc::new(vec![
            PointXYZRGB::new(1.0, 2.0, 3.0, 255),
            PointXYZRGB::new(4.0, 5.0, 6.0, 255),
        ]);
        kdtree.set_input_cloud(cloud.clone(), None);
        let query_point = PointXYZRGBWithId::new(0, PointXYZRGB::new(1.1, 2.1, 3.1, 255));
        let (indices, distances) = kdtree.nearest_k_search(&query_point, 1);
        assert_eq!(indices.len(), 1);
        assert_eq!(distances.len(), 1);
    }

    // 测试 radius_search 方法
    #[test]
    fn test_radius_search() {
        let mut kdtree = KdTreeRust::<PointXYZRGBWithId>::new(true);
        let cloud = Arc::new(vec![
            PointXYZRGB::new(1.0, 2.0, 3.0, 255),
            PointXYZRGB::new(4.0, 5.0, 6.0, 255),
        ]);
        kdtree.set_input_cloud(cloud.clone(), None);
        let query_point = PointXYZRGBWithId::new(0, PointXYZRGB::new(1.1, 2.1, 3.1, 255));
        let (indices, distances) = kdtree.radius_search(&query_point, 1.0, 10);
        assert_eq!(indices.len(), 1);
        assert_eq!(distances.len(), 1);
    }

    // 测试 set_epsilon 和 get_epsilon 方法
    #[test]
    fn test_epsilon() {
        let mut kdtree = KdTreeRust::<PointXYZRGBWithId>::new(true);
        kdtree.set_epsilon(0.1);
        assert_eq!(kdtree.get_epsilon(), 0.1);
    }

    // 测试 set_min_pts 和 get_min_pts 方法
    #[test]
    fn test_min_pts() {
        let mut kdtree = KdTreeRust::<PointXYZRGBWithId>::new(true);
        kdtree.set_min_pts(5);
        assert_eq!(kdtree.get_min_pts(), 5);
    }
}

// 20. 类型: PointXYZRGBNormal
#[cfg(test)]
mod tests20 {
    use super::*;
    use std::sync::Arc;

    // 测试 KdTreeRust<PointXYZRGBNormalWithId> 的构造函数
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
}
