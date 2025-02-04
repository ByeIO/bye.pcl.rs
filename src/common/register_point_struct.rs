#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]

//! 处理点云数据的结构和操作

// 异常处理
use anyhow::Result;
// 浮点数精度处理
use approx;
// 二进制编码
use base64;
// 命令行参数解析
use cpal;
// 性能评测
use criterion;
// kdtree数据结构
use kd_tree;
// 线性代数库
use nalgebra::DMatrix;
// 复数支持
use num_complex;
// 数学特性
use num_traits;
// 宏编程/元编程
use paste;
use proc_macro2;
use quote;
use syn;
// 随机数
use rand::{self, Rng};
// 随机数分布
use rand_distr;
// fft快速傅立叶变换
use realfft;
// 机器学习,数据分析
use rstats;
// 序列化
use serde::{Deserialize, Serialize};
// 多线程
use tokio;

