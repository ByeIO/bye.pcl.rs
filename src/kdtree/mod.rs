//! kd树最近邻搜索

// 1. 实现kdpoint特性
pub mod impl_kdpoint;
pub use impl_kdpoint::*;

// 2. 提供kd树数据结构, 接口名称尽量接近cpp源码
pub mod kdtree_;
// pub use kdtree_::KdTreeRust as KdTree;
pub use kdtree_::KdTreeRust;

// 3. 适配kd树快速搜索算法
pub mod kdtree_flann;
pub use kdtree_flann::*;
