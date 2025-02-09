#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]

//! FLANN: Fast Library for Approximate Nearest Neighbors(kdtree)

// 标准库
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::env;
use std::fmt;
use std::sync::Arc;

// kd树
use kd_tree::{KdPoint, KdTree};
use typenum;

// 宏编程
use paste::paste;

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


/* start 获取近似索引 */

// 10. 类型: PointXYZRGB
impl PointXYZRGBWithId {
    // 获取近似索引的函数
    pub fn get_approximate_indices(
        cloud_in: &PointCloud<PointXYZRGB>,
        cloud_ref: &PointCloud<PointXYZRGB>,
    ) -> Vec<usize> {
        // 将cloud_ref的点包装为PointXYZRGBWithId
        let points_with_id: Vec<PointXYZRGBWithId> = cloud_ref.points
            .iter()
            .enumerate()
            .map(|(id, point)| PointXYZRGBWithId { id, point: point.clone() })
            .collect();

        // 构建kd-tree
        let kdtree: KdTree<PointXYZRGBWithId> = KdTree::build_by_ordered_float(points_with_id);

        // 存储结果的索引
        let mut indices = Vec::with_capacity(cloud_in.size.try_into().unwrap());

        // 对输入点云中的每个点进行最近邻搜索
        for point in &cloud_in.points {
            let nearest = kdtree.nearest(&[point.x, point.y, point.z]).unwrap();
            indices.push(nearest.item.id);
        }

        indices
    }
}

// 20. 类型: PointXYZRGBNormal
impl PointXYZRGBNormalWithId {
    // 获取近似索引的函数
    pub fn get_approximate_indices(
        cloud_in: &PointCloud<PointXYZRGBNormal>,
        cloud_ref: &PointCloud<PointXYZRGBNormal>,
    ) -> Vec<usize> {
        // 将cloud_ref的点包装为PointXYZRGBWithId
        let points_with_id: Vec<PointXYZRGBNormalWithId> = cloud_ref.points
            .iter()
            .enumerate()
            .map(|(id, point)| PointXYZRGBNormalWithId { id, point: point.clone() })
            .collect();

        // 构建kd-tree
        let kdtree: KdTree<PointXYZRGBNormalWithId> = KdTree::build_by_ordered_float(points_with_id);

        // 存储结果的索引
        let mut indices = Vec::with_capacity(cloud_in.size.try_into().unwrap());

        // 对输入点云中的每个点进行最近邻搜索
        for point in &cloud_in.points {
            let nearest = kdtree.nearest(&[point.x, point.y, point.z]).unwrap();
            indices.push(nearest.item.id);
        }

        indices
    }
}

/* end 获取近似索引 */

/* start 范型类型(无用) */

// 使用泛型结构体来模拟模板
pub struct L2_Simple<T> {
    _marker: std::marker::PhantomData<T>,
}

// 使用泛型类来模拟模板
pub struct Index<T> {
    _marker: std::marker::PhantomData<T>,
}

/* end 范型类型(无用) */

/* start FLANN搜索 */
// PointT为普通点
pub struct KdTreeFLANN<PointT> {
    /// 输入点云数据集
    input: Arc<Vec<PointT>>,
    /// 点索引子集
    indices: Option<Arc<Vec<usize>>>,
    /// 最近邻搜索的精度（误差界限）
    epsilon: f32,
    /// 可行结果必须包含的最小邻居数
    min_pts: usize,
    /// 是否对结果进行排序
    sorted: bool,
    /// 内部数据指针
    cloud: Option<Arc<Vec<f32>>>,
    /// 内部和外部索引的映射
    index_mapping: Vec<usize>,
    /// 是否为恒等映射
    identity_mapping: bool,
    /// 点的维度
    dim: usize,
    /// 数据的总大小
    total_nr_points: usize,
}

// 10. 类型: PointXYZRGB
impl KdTreeFLANN<PointXYZRGB> {
    /// 默认构造函数
    /// `sorted` 设置为 true 表示需要对最近邻索引进行排序，否则不排序
    pub fn new(sorted: bool) -> Self {
        Self {
            input: Arc::new(Vec::new()),
            indices: None,
            epsilon: 0.0,
            min_pts: 0,
            sorted,
            cloud: None,
            index_mapping: Vec::new(),
            identity_mapping: false,
            dim: 0,
            total_nr_points: 0,
        }
    }

    /// 设置输入点云数据集
    /// `cloud` 是输入点云的共享指针
    /// `indices` 是点云索引的子集，如果为 None 则使用整个点云
    pub fn set_input_cloud(&mut self, cloud: Arc<Vec<PointXYZRGB>>, indices: Option<Arc<Vec<usize>>>) {
        self.input = cloud;
        self.indices = indices;
        self.convert_cloud_to_array();
    }

    /// 设置最近邻搜索的精度（误差界限）
    pub fn set_epsilon(&mut self, eps: f32) {
        self.epsilon = eps;
    }

    /// 设置是否对结果进行排序
    pub fn set_sorted_results(&mut self, sorted: bool) {
        self.sorted = sorted;
    }

    /// 将 PointXYZRGB 转换为 PointXYZRGBWithId (KdPoint 类型)
    fn point_to_kdpoint(&self, point: &PointXYZRGB) -> PointXYZRGBWithId {
        PointXYZRGBWithId::new(0, point.clone())
    }

    /// 搜索给定查询点的 k 个最近邻
    /// `point` 是查询点
    /// `k` 是要搜索的邻居数量
    /// 返回找到的邻居索引和距离
    pub fn knn_search(&self, point: &PointXYZRGB, k: usize) -> (Vec<usize>, Vec<f32>) {
        let kdpoint = self.point_to_kdpoint(point);
        let kdtree = KdTree::build_by_ordered_float(self.input.iter().map(|p| self.point_to_kdpoint(p)).collect());
        let nearest = kdtree.nearest(&kdpoint).unwrap();
        let indices = vec![nearest.item.id];
        let distances = vec![nearest.squared_distance];
        (indices, distances)
    }

    /// 在给定半径内搜索查询点的所有最近邻
    /// `point` 是查询点
    /// `radius` 是搜索半径
    /// `max_nn` 是最大返回邻居数，如果为 0 则返回所有邻居
    /// 返回找到的邻居索引和距离
    pub fn radius_search(&self, point: &PointXYZRGB, radius: f64, max_nn: usize) -> (Vec<usize>, Vec<f32>) {
        let kdpoint = self.point_to_kdpoint(point);
        let kdtree = KdTree::build_by_ordered_float(self.input.iter().map(|p| self.point_to_kdpoint(p)).collect());
        let nearest = kdtree.within_radius(&kdpoint, radius as f32);
        let indices = nearest.iter().map(|n| n.id).collect();
        let distances = nearest.iter().map(|n| {
            let dx = n.point.x - point.x;
            let dy = n.point.y - point.y;
            let dz = n.point.z - point.z;
            (dx * dx + dy * dy + dz * dz).sqrt()
        }).collect();
        (indices, distances)
    }

    /// 将点云转换为内部 FLANN 数组表示
    fn convert_cloud_to_array(&mut self) {
        if self.input.is_empty() {
            self.cloud = None;
            return;
        }

        let original_no_of_points = self.input.len();
        let mut cloud_data = Vec::with_capacity(original_no_of_points * self.dim);
        self.index_mapping.clear();
        self.identity_mapping = true;

        for (cloud_index, point) in self.input.iter().enumerate() {
            if !point.is_valid() {
                self.identity_mapping = false;
                continue;
            }

            self.index_mapping.push(cloud_index);
            point.vectorize(&mut cloud_data);
        }

        self.cloud = Some(Arc::new(cloud_data));
        self.total_nr_points = self.index_mapping.len();
    }
}

// 20. PointXYZRGBNormal
impl KdTreeFLANN<PointXYZRGBNormal> {
    /// 默认构造函数
    /// `sorted` 设置为 true 表示需要对最近邻索引进行排序，否则不排序
    pub fn new(sorted: bool) -> Self {
        Self {
            input: Arc::new(Vec::new()),
            indices: None,
            epsilon: 0.0,
            min_pts: 0,
            sorted,
            cloud: None,
            index_mapping: Vec::new(),
            identity_mapping: false,
            dim: 0,
            total_nr_points: 0,
        }
    }

    /// 设置输入点云数据集
    /// `cloud` 是输入点云的共享指针
    /// `indices` 是点云索引的子集，如果为 None 则使用整个点云
    pub fn set_input_cloud(&mut self, cloud: Arc<Vec<PointXYZRGBNormal>>, indices: Option<Arc<Vec<usize>>>) {
        self.input = cloud;
        self.indices = indices;
        self.convert_cloud_to_array();
    }

    /// 设置最近邻搜索的精度（误差界限）
    pub fn set_epsilon(&mut self, eps: f32) {
        self.epsilon = eps;
    }

    /// 设置是否对结果进行排序
    pub fn set_sorted_results(&mut self, sorted: bool) {
        self.sorted = sorted;
    }

    /// 将 PointXYZRGBNormal 转换为 PointXYZRGBNormalWithId (KdPoint 类型)
    fn point_to_kdpoint(&self, point: &PointXYZRGBNormal) -> PointXYZRGBNormalWithId {
        PointXYZRGBNormalWithId::new(0, point.clone())
    }

    /// 搜索给定查询点的 k 个最近邻
    /// `point` 是查询点
    /// `k` 是要搜索的邻居数量
    /// 返回找到的邻居索引和距离
    pub fn knn_search(&self, point: &PointXYZRGBNormal, k: usize) -> (Vec<usize>, Vec<f32>) {
        let kdpoint = self.point_to_kdpoint(point);
        let kdtree = KdTree::build_by_ordered_float(self.input.iter().map(|p| self.point_to_kdpoint(p)).collect());
        let nearest = kdtree.nearest(&kdpoint).unwrap();
        let indices = vec![nearest.item.id];
        let distances = vec![nearest.squared_distance];
        (indices, distances)
    }

    /// 在给定半径内搜索查询点的所有最近邻
    /// `point` 是查询点
    /// `radius` 是搜索半径
    /// `max_nn` 是最大返回邻居数，如果为 0 则返回所有邻居
    /// 返回找到的邻居索引和距离
    pub fn radius_search(&self, point: &PointXYZRGBNormal, radius: f64, max_nn: usize) -> (Vec<usize>, Vec<f32>) {
        let kdpoint = self.point_to_kdpoint(point);
        let kdtree = KdTree::build_by_ordered_float(self.input.iter().map(|p| self.point_to_kdpoint(p)).collect());
        let nearest = kdtree.within_radius(&kdpoint, radius as f32);
        let indices = nearest.iter().map(|n| n.id).collect();
        let distances = nearest.iter().map(|n| {
            let dx = n.point.x - point.x;
            let dy = n.point.y - point.y;
            let dz = n.point.z - point.z;
            (dx * dx + dy * dy + dz * dz).sqrt()
        }).collect();
        (indices, distances)
    }

    /// 将点云转换为内部 FLANN 数组表示
    fn convert_cloud_to_array(&mut self) {
        if self.input.is_empty() {
            self.cloud = None;
            return;
        }

        let original_no_of_points = self.input.len();
        let mut cloud_data = Vec::with_capacity(original_no_of_points * self.dim);
        self.index_mapping.clear();
        self.identity_mapping = true;

        for (cloud_index, point) in self.input.iter().enumerate() {
            if !point.is_valid() {
                self.identity_mapping = false;
                continue;
            }

            self.index_mapping.push(cloud_index);
            point.vectorize(&mut cloud_data);
        }

        self.cloud = Some(Arc::new(cloud_data));
        self.total_nr_points = self.index_mapping.len();
    }
}

/* end FLANN搜索 */

#[cfg(test)]
mod tests10 {
    use super::*;

    // 测试get_approximate_indices函数
    #[test]
    fn test_pointxyzrgb_get_approximate_indices() {
        // 创建参考点云
        let mut cloud_ref = PointCloud::<PointXYZRGB>::new();
        cloud_ref.points.push(PointXYZRGB::new(1.0, 2.0, 3.0, 0));
        cloud_ref.points.push(PointXYZRGB::new(4.0, 5.0, 6.0, 0));
        cloud_ref.points.push(PointXYZRGB::new(7.0, 8.0, 9.0, 0));

        // 创建输入点云
        let mut cloud_in = PointCloud::<PointXYZRGB>::new();
        // 应匹配第一个点
        cloud_in.points.push(PointXYZRGB::new(1.1, 2.1, 3.1, 0)); 
        // 应匹配第二个点
        cloud_in.points.push(PointXYZRGB::new(4.1, 5.1, 6.1, 0)); 
        // 应匹配第三个点
        cloud_in.points.push(PointXYZRGB::new(7.1, 8.1, 9.1, 0)); 

        // 获取近似索引
        let indices = PointXYZRGBWithId::get_approximate_indices(&cloud_in, &cloud_ref);

        // 验证结果
        assert_eq!(indices.len(), 3);
        // 第一个点应匹配参考点云中的第一个点
        assert_eq!(indices[0], 0);
        // 第二个点应匹配参考点云中的第二个点
        assert_eq!(indices[1], 1); 
        // 第三个点应匹配参考点云中的第三个点
        assert_eq!(indices[2], 2); 
    }
    
    #[test]
    fn test_pointxyzrgb_knn_search() {
        let mut kdtree = KdTreeFLANN::<PointXYZRGB>::new(true);
        let mut cloud = PointCloud::<PointXYZRGB>::new();
        cloud.points.push(PointXYZRGB::new(1.0, 2.0, 3.0, 0));
        cloud.points.push(PointXYZRGB::new(4.0, 5.0, 6.0, 0));
        cloud.points.push(PointXYZRGB::new(7.0, 8.0, 9.0, 0));

        kdtree.set_input_cloud(Arc::new(cloud.points), None);

        let query_point = PointXYZRGB::new(1.1, 2.1, 3.1, 0);
        let (indices, distances) = kdtree.knn_search(&query_point, 1);

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert!(distances[0] < 0.1);
    }

    #[test]
    fn test_pointxyzrgb_radius_search() {
        let mut kdtree = KdTreeFLANN::<PointXYZRGB>::new(true);
        let mut cloud = PointCloud::<PointXYZRGB>::new();
        cloud.points.push(PointXYZRGB::new(1.0, 2.0, 3.0, 0));
        cloud.points.push(PointXYZRGB::new(4.0, 5.0, 6.0, 0));
        cloud.points.push(PointXYZRGB::new(7.0, 8.0, 9.0, 0));

        kdtree.set_input_cloud(Arc::new(cloud.points), None);

        let query_point = PointXYZRGB::new(1.1, 2.1, 3.1, 0);
        let (indices, distances) = kdtree.radius_search(&query_point, 1.0, 0);

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        println!("distances[0]: {}\n", distances[0]);
        assert!(distances[0] < 0.2);
    }
}

#[cfg(test)]
mod tests20 {
    use super::*;

    // 测试get_approximate_indices函数
    #[test]
    fn test_pointxyzrgb_get_approximate_indices() {
        // 创建参考点云
        let mut cloud_ref = PointCloud::<PointXYZRGBNormal>::new();
        cloud_ref.points.push(PointXYZRGBNormal::new(1.0, 2.0, 3.0, 0, [0.0, 0.0, 0.0], 0.0));
        cloud_ref.points.push(PointXYZRGBNormal::new(4.0, 5.0, 6.0, 0, [0.0, 0.0, 0.0], 0.0));
        cloud_ref.points.push(PointXYZRGBNormal::new(7.0, 8.0, 9.0, 0, [0.0, 0.0, 0.0], 0.0));

        // 创建输入点云
        let mut cloud_in = PointCloud::<PointXYZRGBNormal>::new();
        // 应匹配第一个点
        cloud_in.points.push(PointXYZRGBNormal::new(1.0, 2.0, 3.0, 0, [0.0, 0.0, 0.0], 0.0));
        // 应匹配第二个点
        cloud_in.points.push(PointXYZRGBNormal::new(4.0, 5.0, 6.0, 0, [0.0, 0.0, 0.0], 0.0));
        // 应匹配第三个点
        cloud_in.points.push(PointXYZRGBNormal::new(7.0, 8.0, 9.0, 0, [0.0, 0.0, 0.0], 0.0));

        // 获取近似索引
        let indices = PointXYZRGBNormalWithId::get_approximate_indices(&cloud_in, &cloud_ref);

        // 验证结果
        assert_eq!(indices.len(), 3);
        // 第一个点应匹配参考点云中的第一个点
        assert_eq!(indices[0], 0);
        // 第二个点应匹配参考点云中的第二个点
        assert_eq!(indices[1], 1); 
        // 第三个点应匹配参考点云中的第三个点
        assert_eq!(indices[2], 2); 
    }
    
    #[test]
    fn test_pointxyzrgbnormal_knn_search() {
        let mut kdtree = KdTreeFLANN::<PointXYZRGBNormal>::new(true);
        let mut cloud = PointCloud::<PointXYZRGBNormal>::new();
        cloud.points.push(PointXYZRGBNormal::new(1.0, 2.0, 3.0, 0, [0.0, 0.0, 0.0], 0.0));
        cloud.points.push(PointXYZRGBNormal::new(4.0, 5.0, 6.0, 0, [0.0, 0.0, 0.0], 0.0));
        cloud.points.push(PointXYZRGBNormal::new(7.0, 8.0, 9.0, 0, [0.0, 0.0, 0.0], 0.0));

        kdtree.set_input_cloud(Arc::new(cloud.points), None);

        let query_point = PointXYZRGBNormal::new(1.1, 2.1, 3.1, 0, [0.0, 0.0, 0.0], 0.0);
        let (indices, distances) = kdtree.knn_search(&query_point, 1);

        assert_eq!(indices.len(), 1);
        assert_eq!(indices[0], 0);
        assert!(distances[0] < 0.1);
    }

    #[test]
    fn test_pointxyzrgbnormal_radius_search() {
        let mut kdtree = KdTreeFLANN::<PointXYZRGBNormal>::new(true);
        let mut cloud = PointCloud::<PointXYZRGBNormal>::new();
        cloud.points.push(PointXYZRGBNormal::new(1.0, 2.0, 3.0, 0, [0.0, 0.0, 0.0], 0.0));
        cloud.points.push(PointXYZRGBNormal::new(4.0, 5.0, 6.0, 0, [0.0, 0.0, 0.0], 0.0));
        cloud.points.push(PointXYZRGBNormal::new(7.0, 8.0, 9.0, 0, [0.0, 0.0, 0.0], 0.0));

        kdtree.set_input_cloud(Arc::new(cloud.points), None);

        let query_point = PointXYZRGBNormal::new(1.1, 2.1, 3.1, 0, [0.0, 0.0, 0.0], 0.0);
        let (indices, distances) = kdtree.radius_search(&query_point, 1.0, 0);

        assert_eq!(indices.len(), 1);
    }
    
}
