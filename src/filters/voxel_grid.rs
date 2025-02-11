#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]

//! 体素网格滤波器

// 标准库
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::env;
use std::fmt::Debug;
use std::collections::HashMap;

// 宏编程
use paste::paste;

// f3l库
use f3l_filter::F3lFilterInverse;
use f3l_filter::F3lFilter;
use f3l_core::serde::{self, Deserialize, Serialize};
use f3l_core::{get_minmax, BasicFloat};

// 适配f3l库的体素网格滤波器算法
// 原作: authors = ["Donvlouss"] 
// 许可: license = "MIT OR Apache-2.0"

/// 构建一个“按维度划分”的网格，并计算每个网格内点的平均值。
/// 
/// # 示例
/// ```rust
/// let vertices = load_ply("../../data/table_scene_lms400.ply");
/// 
/// let mut filter = VoxelGrid::with_data(&[0.05; 3]);
/// use std::time::Instant;
/// let start = Instant::now();
/// 
/// let out = filter.filter_instance(&vertices);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(crate = "self::serde")]
pub struct VoxelGridParameter<T: BasicFloat, const D: usize> {
    // 边界范围，每个元素是一个元组，表示一个维度的最小值和最大值
    pub bound: Vec<(T, T)>,
    // 每个维度的逆分割因子
    pub inverse_div: Vec<T>,
    // 每个维度的网格数量
    pub nb_dim: Vec<usize>,
}

impl<T: BasicFloat, const D: usize> VoxelGridParameter<T, D> {
    /// 创建一个新的 `VoxelGridParameter` 实例
    pub fn new() -> Self {
        Self {
            bound: Vec::with_capacity(D),
            inverse_div: Vec::with_capacity(D),
            nb_dim: Vec::with_capacity(D),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "self::serde")]
pub struct VoxelGrid<T: BasicFloat, const D: usize> {
    // 每个维度的网格大小
    pub leaf: Vec<T>,
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    // 体素网格的参数
    pub parameter: VoxelGridParameter<T, D>,
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    // 体素映射，键是体素索引，值是该体素内点的索引列表
    pub voxel_map: HashMap<usize, Vec<usize>>,
}

impl<T: BasicFloat, const D: usize> Default for VoxelGrid<T, D> {
    /// 返回 `VoxelGrid` 的默认实例
    fn default() -> Self {
        Self::new()
    }
}

impl<T: BasicFloat, const D: usize> VoxelGrid<T, D> {
    /// 创建一个新的 `VoxelGrid` 实例
    pub fn new() -> Self {
        Self {
            leaf: vec![T::zero(); D],
            parameter: VoxelGridParameter::new(),
            voxel_map: HashMap::new(),
        }
    }

    /// 使用给定的网格大小创建一个新的 `VoxelGrid` 实例
    pub fn with_data<P: Into<[T; D]> + Copy>(leaf: &P) -> Self {
        let leaf: [T; D] = (*leaf).into();
        Self {
            leaf: leaf.iter().copied().collect(),
            parameter: VoxelGridParameter::<T, D>::new(),
            voxel_map: HashMap::new(),
        }
    }

    /// 设置网格大小
    pub fn set_leaf<P: Into<[T; D]> + Copy>(&mut self, leaf: &P) {
        let leaf: [T; D] = (*leaf).into();
        self.leaf = leaf.iter().copied().collect();
    }

    /// 直接设置网格大小
    pub fn set_leaf_raw(&mut self, leaf: &[T; D]) {
        (0..D).for_each(|i| self.leaf[i] = leaf[i]);
    }

    /// 计算数据的边界范围
    pub fn compute_bound<P: Into<[T; D]> + Copy>(&mut self, data: &[P])
    where
        [T; D]: Into<P>,
    {
        if data.is_empty() {
            return;
        }
        let (min, max) = get_minmax(data);
        let (min, max): ([T; D], [T; D]) = (min.into(), max.into());
        self.parameter.bound = min.into_iter().zip(max).collect();
    }

    /// 检查网格设置是否有效
    pub fn leaf_check<P: Into<[T; D]> + Copy>(&mut self, data: &[P]) -> bool
    where
        [T; D]: Into<P>,
    {
        if self.parameter.bound.is_empty() {
            self.compute_bound(data);
        }
        let mut inverse = vec![T::zero(); D];
        let bounds = &self.parameter.bound;
        let mut box_range = [T::zero(); D];
        (0..D).for_each(|i| {
            let dif = bounds[i].1 - bounds[i].0;
            box_range[i] = dif / self.leaf[i];
            inverse[i] = T::one() / self.leaf[i];
        });

        let mut dims = vec![0usize; D];
        let mut count: usize = 0;
        // 检查体素数量是否会溢出
        for (i, v) in box_range.iter().enumerate() {
            match v.to_usize() {
                Some(v) => {
                    // 加 1 以避免值过小
                    dims[i] = v + 1;
                    if let Some(vv) = count.checked_mul(v + 1) {
                        count = vv;
                    } else {
                        return false;
                    }
                }
                None => return false,
            }
        }
        self.parameter.inverse_div = inverse;
        self.parameter.nb_dim = dims;

        true
    }
}

impl<T: BasicFloat, const D: usize> F3lFilterInverse for VoxelGrid<T, D> {
    /// 设置是否使用负过滤（这里不做实际操作）
    fn set_negative(&mut self, _negative: bool) {}
}

impl<'a, P, T: BasicFloat, const D: usize> F3lFilter<'a, P, D> for VoxelGrid<T, D>
where
    P: Into<[T; D]> + Clone + Copy + Send + Sync + Debug,
    [T; D]: Into<P>,
{
    /// 获取非空体素网格的索引（不是点的索引）
    fn filter(&mut self, data: &'a Vec<P>) -> Vec<usize> {
        if !self.apply_filter(data) {
            return vec![];
        }

        let keys = self.voxel_map.keys();
        keys.into_iter().copied().collect()
    }

    /// 获取非空网格的中心点
    fn filter_instance(&mut self, data: &'a Vec<P>) -> Vec<P> {
        if !self.apply_filter(data) {
            return vec![];
        }

        let maps = &self.voxel_map;
        maps.iter()
            .map(|(_, pts)| {
                let nb = T::from(pts.len()).unwrap();
                let factor = T::one() / nb;
                let mut sum = [T::zero(); D];
                pts.iter().for_each(|p| {
                    let p: [T; D] = data[*p].into();
                    (0..D).for_each(|i| {
                        sum[i] += p[i] * factor;
                    });
                });
                sum.into()
            })
            .collect()
    }

    /// 应用过滤操作
    fn apply_filter(&mut self, data: &'a Vec<P>) -> bool {
        if !self.leaf_check(data) {
            return false;
        }

        let VoxelGridParameter {
            bound,
            inverse_div: inv_div,
            nb_dim,
        } = &self.parameter;

        let min = bound.iter().map(|(a, _)| *a).collect::<Vec<_>>();

        let mut inc = [1usize; D];
        (0..D).for_each(|i| {
            if i == 0 {
                return;
            }

            inc[i] = nb_dim[i - 1] * inc[i - 1];
        });

        data.iter().enumerate().for_each(|(i, p)| {
            let p: [T; D] = (*p).into();
            let mut dim = 0usize;
            (0..D).for_each(|i| {
                let v = (p[i] - min[i]) * inv_div[i];
                let d = match v.to_usize() {
                    Some(d) => d,
                    None => return,
                };
                dim += d * inc[i];
            });
            let vec = self.voxel_map.entry(dim).or_default();
            vec.push(i);
        });
        true
    }
}

#[test]
fn serde() {
    let model = VoxelGrid::with_data(&[0.05f32; 3]);
    let content = serde_json::to_string(&model).unwrap();
    println!("{}", content);
    let text = r#"{"leaf":[0.05,0.05,0.05]}"#;
    let model_de: VoxelGrid<f32, 3> = serde_json::from_str(text).unwrap();
    assert_eq!(model.leaf, model_de.leaf);
}
