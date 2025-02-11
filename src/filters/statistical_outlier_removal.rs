#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]

//! 统计异常值移除滤波器

// 标准库
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::env;
use std::fmt;
use std::ops::Index;

// 宏编程
use paste::paste;

// f3l库(3D点云库)
use f3l_filter::{F3lFilter, F3lFilterInverse};
use f3l_core::rayon::prelude::*;
use f3l_core::{
    serde::{self, Deserialize, Serialize},
    BasicFloat,
};
use f3l_search_tree::{KdTree, TreeSearch};

// 适配f3l库的统计异常值移除滤波器算法
// 原作: authors = ["Donvlouss"] 
// 许可: license = "MIT OR Apache-2.0"

/// 使用点邻域统计来过滤异常数据
/// 该算法通过两次迭代输入数据：
/// 第一次迭代计算每个点到其最近的k个邻居的平均距离
/// 然后计算所有这些距离的均值和标准差，以确定距离阈值
/// 距离阈值等于：均值 + stddev_mult * 标准差
/// 在下一次迭代中，如果平均邻居距离低于或高于该阈值，则将点分类为内点或外点
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "self::serde")]
pub struct StatisticalOutlierRemoval<'a, P, T: BasicFloat, const D: usize>
where
    P: Into<[T; D]> + Clone + Copy + Index<usize, Output = T>,
{
    pub negative: bool, // 是否提取被移除的点的索引
    pub multiply: T, // 标准差乘数
    pub k_neighbors: usize, // 最近邻居的数量
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    pub tree: KdTree<'a, T, P>, // Kd树
    #[serde(skip_serializing)]
    #[serde(skip_deserializing)]
    pub inlier: Vec<bool>, // 内点标记
}

impl<'a, P, T: BasicFloat, const D: usize> StatisticalOutlierRemoval<'a, P, T, D>
where
    P: Into<[T; D]> + Clone + Copy + Send + Sync + Index<usize, Output = T>,
    [T; D]: Into<P>,
{
    /// 构造函数
    pub fn new(multiply: T, k_neighbors: usize) -> Self {
        Self {
            negative: false,
            multiply,
            k_neighbors,
            tree: KdTree::<T, P>::new(D),
            inlier: vec![],
        }
    }

    #[inline]
    pub fn ok(&self, is_inlier: bool) -> bool {
        (!is_inlier && self.negative) || (is_inlier && !self.negative)
    }
}

impl<'a, P, T: BasicFloat, const D: usize> F3lFilterInverse
    for StatisticalOutlierRemoval<'a, P, T, D>
where
    P: Into<[T; D]> + Clone + Copy + Send + Sync + Index<usize, Output = T>,
    [T; D]: Into<P>,
{
    /// 设置是否提取被移除的点的索引
    fn set_negative(&mut self, negative: bool) {
        self.negative = negative;
    }
}

impl<'a, P, T: BasicFloat, const D: usize> F3lFilter<'a, P, D>
    for StatisticalOutlierRemoval<'a, P, T, D>
where
    P: Into<[T; D]> + Clone + Copy + Send + Sync + Index<usize, Output = T>,
    [T; D]: Into<P>,
{
    /// 过滤数据并返回内点的索引
    fn filter(&mut self, data: &'a Vec<P>) -> Vec<usize> {
        self.apply_filter(data);

        self.inlier
            .iter()
            .enumerate()
            .filter(|&(_, f)| self.ok(*f))
            .map(|(i, _)| i)
            .collect()
    }

    /// 过滤数据并返回内点
    fn filter_instance(&mut self, data: &'a Vec<P>) -> Vec<P> {
        self.apply_filter(data);

        self.inlier
            .iter()
            .enumerate()
            .filter(|&(_, f)| self.ok(*f))
            .map(|(i, _)| data[i])
            .collect()
    }

    /// 应用过滤器
    fn apply_filter(&mut self, data: &'a Vec<P>) -> bool {
        if data.is_empty() {
            return false;
        }
        // 检查树的维度是否正确
        if self.tree.dim != D {
            self.tree = KdTree::<T, P>::new(D);
        }
        self.tree.set_data(data);
        self.tree.build();

        use std::sync::{Arc, Mutex};
        let nb_valid = Arc::new(Mutex::new(0usize));

        let distances = data
            .par_iter()
            .enumerate()
            .map(|(i, v)| {
                let out = self.tree.search_knn(v, self.k_neighbors);
                if out.is_empty() {
                    return (i, T::zero());
                }
                {
                    let mut lock = nb_valid.lock().unwrap();
                    *lock += 1usize;
                }

                let sum = out.iter().map(|(_, o)| (*o) * (*o)).sum::<f32>();
                (i, T::from(sum).unwrap())
            })
            .collect::<Vec<_>>();
        let nb_valid = *(nb_valid.lock().unwrap());
        let nb_valid_t = T::from(nb_valid).unwrap();

        let (sum, sq_sum) = distances
            .iter()
            .fold((T::zero(), T::zero()), |(sum, sq_sum), &(_, d)| {
                (sum + d, sq_sum + d * d)
            });
        let mean = sum / nb_valid_t;
        let variance = (sq_sum - sum * sum / nb_valid_t) / (nb_valid_t - T::one());
        let std_dev = variance.sqrt();

        let threshold = mean + self.multiply * std_dev;

        self.inlier = vec![false; data.len()];
        distances.iter().for_each(|&(i, d)| {
            if d <= threshold {
                self.inlier[i] = true;
            }
        });

        true
    }
}

#[test]
fn serde() {
    let model = StatisticalOutlierRemoval::<[f32; 3], f32, 3>::new(2f32, 20_usize);
    let content = serde_json::to_string(&model).unwrap();
    println!("{}", content);

    let text = r#"{
        "negative":false,
        "multiply":2.0,
        "k_neighbors":20
    }"#;
    let model_de: StatisticalOutlierRemoval<[f32; 3], f32, 3> = serde_json::from_str(text).unwrap();
    assert_eq!(model.negative, model_de.negative);
    assert_eq!(model.multiply, model_de.multiply);
    assert_eq!(model.k_neighbors, model_de.k_neighbors);
}
