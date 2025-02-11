#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]

//! 贪婪投影三角化

// 标准库
use std::collections::{ HashMap, BTreeMap };
use std::sync::Arc;
use std::f64::consts::PI;
use std::vec::Vec;
use std::marker::PhantomData;
use std::cell::RefCell;
use std::borrow::{BorrowMut, Borrow};

// 随机数
use rand::Rng;
use rand_distr::Uniform;

// 元编程
use paste::paste;

// kd树
use kd_tree::KdPoint;

// 线性代数
use nalgebra::{
    Vector2, Vector3, Vector4, 
    Matrix2, Matrix3, Matrix, Vector, Const,
    DVector, DMatrix, DefaultAllocator,
    allocator::Allocator, DimName, Dim, 
    storage::Storage, U3, 
};

// 内部库
use crate::common::{
    PointXYZRGB, PointXYZRGBNormal, Normal, PointCloud,
};
use crate::kdtree::kdtree_::KdTreeRust;
use crate::kdtree::PointXYZRGBNormalWithId;

/* start 枚举 */

// GP3类型枚举
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum GP3Type{ 
  NONE = -1,   
  FREE = 0,    
  FRINGE = 1,  
  BOUNDARY = 2,
  COMPLETED = 3,
  ACTIVE = 4,
}

/* end 枚举 */

/* start 结构体 */
// 1. 定义一个用于表示顶点的结构体
#[derive(Clone, Debug, Default)]
pub struct Vertices {
    pub vertices: Vec<usize>,
}

// 2. 定义一个用于表示多边形网格的结构体
#[derive(Clone, Debug, Default)]
pub struct PolygonMesh {
    pub polygons: Vec<Vertices>,
}

// 3. 定义一个用于存储到最近邻点角度的结构体
#[derive(Clone, Debug, Default)]
pub struct NNAngle {
    pub angle: f64,
    pub index: usize,
    pub nn_index: usize,
    pub visible: bool,
}

// 4. 定义一个用于存储从边缘点出发的边的结构体
#[derive(Clone, Debug, Default)]
pub struct DoubleEdge {
    pub index: usize,
    pub first: Vector2<f64>,
    pub second: Vector2<f64>,
}

// 5. 定义 GreedyProjectionTriangulation 结构体
#[derive(Clone, Debug, Default)]
pub struct GreedyProjectionTriangulation<PointInT: kd_tree::KdPoint> {
    // 最近邻距离乘数，用于获得最终搜索半径
    pub mu: f64,
    // 每个点的最近邻搜索半径和最大边长
    pub search_radius: f64,
    // 搜索接受的最大最近邻数量
    pub nnn: usize,
    // 三角形的首选最小角度
    pub minimum_angle: f64,
    // 三角形的最大角度
    pub maximum_angle: f64,
    // 最大表面角度
    pub eps_angle: f64,
    // 输入法线是否一致定向的标志
    pub consistent: bool,
    // 输出三角形顶点是否应一致定向的标志
    pub consistent_ordering: bool,

    // 临时变量，用于存储三角形（作为点索引集）
    pub triangle: Vertices,
    // 临时变量，用于存储点坐标
    pub coords: Vec<Vector3<f64>>,
    // 到邻居的角度列表
    pub angles: Vec<NNAngle>,
    // 当前查询点的索引
    pub r: usize,
    // 点状态列表
    pub state: Vec<i32>,
    // 源列表
    pub source: Vec<usize>,
    // 一个方向上的边缘邻居列表
    pub ffn: Vec<usize>,
    // 另一个方向上的边缘邻居列表
    pub sfn: Vec<usize>,
    // 每个点的连通分量标签
    pub part: Vec<i32>,
    // 网格必须从其生长的外边缘上的点
    pub fringe_queue: Vec<usize>,

    // 当前点是否为自由点的标志
    pub is_current_free: bool,
    // 当前点的索引
    pub current_index: usize,
    // 前一个点是否为第一个边缘邻居的标志
    pub prev_is_ffn: bool,
    // 前一个点是否为第二个边缘邻居的标志
    pub prev_is_sfn: bool,
    // 下一个点是否为第一个边缘邻居的标志
    pub next_is_ffn: bool,
    // 下一个点是否为第二个边缘邻居的标志
    pub next_is_sfn: bool,
    // 第一个边缘邻居是否被更改的标志
    pub changed_1st_fn: bool,
    // 第二个边缘邻居是否被更改的标志
    pub changed_2nd_fn: bool,
    // 新的边界点
    pub new2boundary: usize,

    // 下一个邻居是否在先前步骤中已连接的标志
    pub already_connected: bool,

    // 点坐标投影到由点法线定义的平面上
    pub proj_qp: Vector3<f64>,
    // 2D 坐标系的第一个坐标向量
    pub u: Vector3<f64>,
    // 2D 坐标系的第二个坐标向量
    pub v: Vector3<f64>,
    // 第一个边缘邻居的 2D 坐标
    pub uvn_ffn: Vector2<f64>,
    // 第二个边缘邻居的 2D 坐标
    pub uvn_sfn: Vector2<f64>,
    // 下一个点的第一个边缘邻居的 2D 坐标
    pub uvn_next_ffn: Vector2<f64>,
    // 下一个点的第二个边缘邻居的 2D 坐标
    pub uvn_next_sfn: Vector2<f64>,

    // 临时变量，用于存储 3 个坐标
    pub tmp: Vector3<f64>,

    // 输入点云
    pub input: Option<PointCloud<PointInT>>,
    // 点索引
    pub indices: Option<Vec<usize>>,
    // KD 树
    pub kdtree: KdTreeRust<PointInT>,
}
/* end 结构体 */

/* start 辅助函数 */
// 判断点 X 是否从点 R（或原点）可见，考虑点 S1 和 S2 之间的线段
fn is_visible(X: &Vector2<f64>, S1: &Vector2<f64>, S2: &Vector2<f64>, R: &Vector2<f64> ) -> bool {
    let a0 = S1[1] - S2[1];
    let b0 = S2[0] - S1[0];
    let c0 = S1[0] * S2[1] - S2[0] * S1[1];
    let mut a1 = -X[1];
    let mut b1 = X[0];
    let mut c1 = 0.0;
    if *R != Vector2::zeros() {
        a1 += R[1];
        b1 -= R[0];
        c1 = R[0] * X[1] - X[0] * R[1];
    }
    let div = a0 * b1 - b0 * a1;
    let x = (b0 * c1 - b1 * c0) / div;
    let y = (a1 * c0 - a0 * c1) / div;

    let mut intersection_outside_XR = false;
    if *R == Vector2::zeros() {
        if X[0] > 0.0 {
            intersection_outside_XR = (x <= 0.0) || (x >= X[0]);
        } else if X[0] < 0.0 {
            intersection_outside_XR = (x >= 0.0) || (x <= X[0]);
        } else if X[1] > 0.0 {
            intersection_outside_XR = (y <= 0.0) || (y >= X[1]);
        } else if X[1] < 0.0 {
            intersection_outside_XR = (y >= 0.0) || (y <= X[1]);
        } else {
            intersection_outside_XR = true;
        }
    } else {
        if X[0] > R[0] {
            intersection_outside_XR = (x <= R[0]) || (x >= X[0]);
        } else if X[0] < R[0] {
            intersection_outside_XR = (x >= R[0]) || (x <= X[0]);
        } else if X[1] > R[1] {
            intersection_outside_XR = (y <= R[1]) || (y >= X[1]);
        } else if X[1] < R[1] {
            intersection_outside_XR = (y >= R[1]) || (y <= X[1]);
        } else {
            intersection_outside_XR = true;
        }
    }
    if intersection_outside_XR {
        return true;
    }
    if S1[0] > S2[0] {
        return (x <= S2[0]) || (x >= S1[0]);
    }
    if S1[0] < S2[0] {
        return (x >= S2[0]) || (x <= S1[0]);
    }
    if S1[1] > S2[1] {
        return (y <= S2[1]) || (y >= S1[1]);
    }
    if S1[1] < S2[1] {
        return (y >= S2[1]) || (y <= S1[1]);
    }
    false
}
/* end 辅助函数 */

/* start 实现 */

// 5. 贪婪投影三角化
impl GreedyProjectionTriangulation<PointXYZRGBNormalWithId> {
    // 5.1 空构造函数
    pub fn new() -> Self {
        Self {
            mu: 0.0,
            search_radius: 0.0,
            nnn: 100,
            minimum_angle: PI / 18.0,
            maximum_angle: 2.0 * PI / 3.0,
            eps_angle: PI / 4.0,
            consistent: false,
            consistent_ordering: false,
            triangle: Vertices { vertices: Vec::new() },
            coords: Vec::new(),
            angles: Vec::new(),
            r: 0,
            state: Vec::new(),
            source: Vec::new(),
            ffn: Vec::new(),
            sfn: Vec::new(),
            part: Vec::new(),
            fringe_queue: Vec::new(),
            is_current_free: false,
            current_index: 0,
            prev_is_ffn: false,
            prev_is_sfn: false,
            next_is_ffn: false,
            next_is_sfn: false,
            changed_1st_fn: false,
            changed_2nd_fn: false,
            new2boundary: 0,
            already_connected: false,
            proj_qp: Vector3::zeros(),
            u: Vector3::zeros(),
            v: Vector3::zeros(),
            uvn_ffn: Vector2::zeros(),
            uvn_sfn: Vector2::zeros(),
            uvn_next_ffn: Vector2::zeros(),
            uvn_next_sfn: Vector2::zeros(),
            tmp: Vector3::zeros(),
            input: None,
            indices: None,
            kdtree: KdTreeRust::<PointXYZRGBNormalWithId>::new(true),
        }
    }

    // 5.2 设置最近邻距离乘数
    pub fn set_mu(&mut self, mu: f64) {
        self.mu = mu;
    }

    // 5.3 获取最近邻距离乘数
    pub fn get_mu(&self) -> f64 {
        self.mu
    }

    // 5.4 设置最大最近邻数量
    pub fn set_maximum_nearest_neighbors(&mut self, nnn: usize) {
        self.nnn = nnn;
    }

    // 5.5 获取最大最近邻数量
    pub fn get_maximum_nearest_neighbors(&self) -> usize {
        self.nnn
    }

    // 5.6 设置搜索半径
    pub fn set_search_radius(&mut self, search_radius: f64) {
        self.search_radius = search_radius;
    }

    // 5.7 获取搜索半径
    pub fn get_search_radius(&self) -> f64 {
        self.search_radius
    }

    // 5.8 设置三角形的最小角度
    pub fn set_minimum_angle(&mut self, minimum_angle: f64) {
        self.minimum_angle = minimum_angle;
    }

    // 5.9 获取三角形的最小角度
    pub fn get_minimum_angle(&self) -> f64 {
        self.minimum_angle
    }

    // 5.10 设置三角形的最大角度
    pub fn set_maximum_angle(&mut self, maximum_angle: f64) {
        self.maximum_angle = maximum_angle;
    }

    // 5.11 获取三角形的最大角度
    pub fn get_maximum_angle(&self) -> f64 {
        self.maximum_angle
    }

    // 5.12 设置最大表面角度
    pub fn set_maximum_surface_angle(&mut self, eps_angle: f64) {
        self.eps_angle = eps_angle;
    }

    // 5.13 获取最大表面角度
    pub fn get_maximum_surface_angle(&self) -> f64 {
        self.eps_angle
    }

    // 5.14 设置输入法线是否一致定向的标志
    pub fn set_normal_consistency(&mut self, consistent: bool) {
        self.consistent = consistent;
    }

    // 5.15 获取输入法线是否一致定向的标志
    pub fn get_normal_consistency(&self) -> bool {
        self.consistent
    }

    // 5.16 设置输出三角形顶点是否应一致定向的标志
    pub fn set_consistent_vertex_ordering(&mut self, consistent_ordering: bool) {
        self.consistent_ordering = consistent_ordering;
    }

    // 5.17 获取输出三角形顶点是否应一致定向的标志
    pub fn get_consistent_vertex_ordering(&self) -> bool {
        self.consistent_ordering
    }

    // 5.18 获取每个点重建后的状态
    pub fn get_point_states(&self) -> Vec<i32> {
        self.state.clone()
    }

    // 5.19 获取每个点重建后的 ID
    pub fn get_part_ids(&self) -> Vec<i32> {
        self.part.clone()
    }

    // 5.20 获取 sfn 列表
    pub fn get_sfn(&self) -> Vec<usize> {
        self.sfn.clone()
    }

    // 5.21 获取 ffn 列表
    pub fn get_ffn(&self) -> Vec<usize> {
        self.ffn.clone()
    }

    // 5.22 执行表面重建，输出多边形网格
    pub fn perform_reconstruction(&mut self, output: &mut PolygonMesh) {
        output.polygons.clear();
        output.polygons.reserve(2 * self.indices.as_ref().unwrap().len());
        if !self.reconstruct_polygons(&mut output.polygons) {
            eprintln!("[GreedyProjectionTriangulation::performReconstruction] Reconstruction failed. Check parameters: search radius ({}) or mu ({}) before continuing.", self.search_radius, self.mu);
            return;
        }
    }

    // 5.23 执行表面重建，输出多边形列表
    pub fn perform_reconstruction_polygons(&mut self, polygons: &mut Vec<Vertices>) {
        polygons.clear();
        polygons.reserve(2 * self.indices.as_ref().unwrap().len());
        if !self.reconstruct_polygons(polygons) {
            eprintln!("[GreedyProjectionTriangulation::performReconstruction] Reconstruction failed. Check parameters: search radius ({}) or mu ({}) before continuing.", self.search_radius, self.mu);
            return;
        }
    }
    
    // 5.24 实际的多边形重建方法
    fn reconstruct_polygons(&mut self, polygons: &mut Vec<Vertices>) -> bool {
        if self.search_radius <= 0.0 || self.mu <= 0.0 {
            polygons.clear();
            return false;
        }
        let sqr_mu = self.mu * self.mu;
        let sqr_max_edge = self.search_radius * self.search_radius;
        if self.nnn > self.indices.as_ref().unwrap().len() {
            self.nnn = self.indices.as_ref().unwrap().len();
        }
    
        let mut nn_idx = vec![0; self.nnn];
        let mut sqr_dists = vec![0.0; self.nnn];
    
        let mut part_index = 0;
    
        let uvn_nn_qp_zero : Vector2<f64> = Vector2::zeros();
        let mut uvn_current : Vector2<f64> = Vector2::zeros();
        let mut uvn_prev : Vector2<f64> = Vector2::zeros();
        let mut uvn_next : Vector2<f64> = Vector2::zeros();
    
        self.already_connected = false;
    
        self.part.clear();
        self.state.clear();
        self.source.clear();
        self.ffn.clear();
        self.sfn.clear();
        self.part.resize(self.indices.as_ref().unwrap().len(), -1);
        self.state.resize(self.indices.as_ref().unwrap().len(), GP3Type::FREE as i32);
        self.source.resize(self.indices.as_ref().unwrap().len(), GP3Type::NONE as i32 as usize);
        self.ffn.resize(self.indices.as_ref().unwrap().len(), GP3Type::NONE as i32 as usize);
        self.sfn.resize(self.indices.as_ref().unwrap().len(), GP3Type::NONE as i32 as usize);
        self.fringe_queue.clear();
        let mut fq_idx = 0;
    
        if let Some(input) = &self.input {
            if !input.is_dense {
                for idx in self.indices.as_ref().unwrap().iter() {
                    let point = &input.points[*idx];
                    if !point.point.x.is_finite() || !point.point.y.is_finite() || !point.point.z.is_finite() {
                        self.state[*idx] = GP3Type::NONE as i32;
                    }
                }
            }
        }
    
        self.coords.clear();
        self.coords.reserve(self.indices.as_ref().unwrap().len());
        let mut point2index = vec![-1; self.input.as_ref().unwrap().points.len()];
        for cp in 0..self.indices.as_ref().unwrap().len() {
            let idx = self.indices.as_ref().unwrap()[cp];
            if let Some(input) = &self.input {
                self.coords.push(Vector3::new(input.points[idx].point.x as f64, input.points[idx].point.y as f64, input.points[idx].point.z as f64));
            }
            point2index[idx] = cp as i32;
        }
    
        let mut is_free = GP3Type::FREE as i32;
        let mut nr_parts = 0;
        let mut increase_nnn4fn = 0;
        let mut increase_nnn4s = 0;
        let mut increase_dist = 0;
        self.angles.resize(self.nnn, NNAngle::default());
        let mut uvn_nn : Vec<Vector2<f64>> = vec![Vector2::zeros(); self.nnn];
        let mut uvn_s : Vector2<f64> = Vector2::zeros();
    
        while is_free != GP3Type::NONE as i32 {
            self.r = is_free as usize;
            if self.state[self.r] == GP3Type::FREE as i32 {
                self.state[self.r] = GP3Type::ACTIVE as i32;
                self.part[self.r] = part_index;
    
                let point = &self.input.as_ref().unwrap().points[self.indices.as_ref().unwrap()[self.r]];
                let mut search_radius = self.search_radius;
                let mut nnn = self.nnn;
                let mut num_neighbors = self.kdtree.radius_search(&point, search_radius, nnn);
    
                
                let mut is_free = GP3Type::FREE as i32;
                let mut nr_parts = 0;
                let mut increase_nnn4fn = 0;
                let mut increase_nnn4s = 0;
                let mut increase_dist = 0;
                self.angles.resize(self.nnn, NNAngle::default());
                let mut uvn_nn: Vec<Vector2<f64>> = vec![Vector2::zeros(); self.nnn];
                let mut uvn_s: Vector2<f64> = Vector2::zeros();
                
                /* start 补充逻辑 */
                while is_free != GP3Type::NONE as i32 {
                    self.r = is_free as usize;
                    if self.state[self.r] == GP3Type::FREE as i32 {
                        self.state[self.r] = GP3Type::ACTIVE as i32;
                        self.part[self.r] = part_index;
                
                        // 创建初始三角形
                        let point = &self.input.as_ref().unwrap().points[self.indices.as_ref().unwrap()[self.r]];
                        let mut search_radius = self.search_radius;
                        let mut nnn = self.nnn;
                        let (nn_idx_vec, sqr_dists_vec) = self.kdtree.radius_search(&point, search_radius, nnn);
                        let num_neighbors = nn_idx_vec.len();
                
                        // 搜索最近邻点
                        let mut nn_idx = vec![0; nnn];
                        let mut sqr_dists = vec![0.0; nnn];
                        for i in 0..num_neighbors {
                            nn_idx[i] = self.indices.as_ref().unwrap()[nn_idx_vec[i]];
                            sqr_dists[i] = sqr_dists_vec[i];
                        }
                
                        let sqr_dist_threshold = sqr_mu.min(sqr_dists[1].into());
                
                        // 获取当前点的法线估计
                        let nc = Vector3::new(
                            point.point.normal[0] as f64,
                            point.point.normal[1] as f64,
                            point.point.normal[2] as f64,
                        );
                
                        // 构建坐标系
                        let mut v = if nc.x.abs() > nc.y.abs() {
                            Vector3::new(-nc.z, 0.0, nc.x).normalize()
                        } else {
                            Vector3::new(0.0, nc.z, -nc.y).normalize()
                        };
                        let u = nc.cross(&v);
                
                        // 投影点到表面
                        let dist = nc.dot(&self.coords[self.r]);
                        let proj_qp = self.coords[self.r] - dist * nc;
                
                        // 转换坐标并计算角度
                        for i in 1..nnn {
                            let tmp = self.coords[nn_idx[i]] - proj_qp;
                            uvn_nn[i][0] = u.dot(&tmp);
                            uvn_nn[i][1] = v.dot(&tmp);
                
                            self.angles[i].angle = uvn_nn[i][1].atan2(uvn_nn[i][0]);
                            self.angles[i].index = nn_idx[i];
                            self.angles[i].visible = true;
                
                            if self.state[nn_idx[i]] == GP3Type::COMPLETED as i32
                                || self.state[nn_idx[i]] == GP3Type::BOUNDARY as i32
                                || self.state[nn_idx[i]] == GP3Type::NONE as i32
                                || sqr_dists[i] > sqr_dist_threshold as f32
                            {
                                self.angles[i].visible = false;
                            }
                        }
                
                        // 排序角度
                        self.angles.sort_by(|a, b| a.angle.partial_cmp(&b.angle).unwrap());
                
                        // 选择第一个可见的自由邻居
                        let mut left = 1;
                        while left < nnn && !self.angles[left].visible {
                            left += 1;
                        }
                
                        if left >= nnn {
                            self.state[self.r] = GP3Type::BOUNDARY as i32;
                            continue;
                        }
                
                        let mut right = left + 1;
                        while right < nnn && !self.angles[right].visible {
                            right += 1;
                        }
                
                        if right >= nnn {
                            self.state[self.r] = GP3Type::BOUNDARY as i32;
                            continue;
                        }
                
                        // 添加三角形
                        self.add_triangle(self.r, self.angles[left].index, self.angles[right].index, polygons);
                
                        // 更新状态
                        self.state[self.r] = GP3Type::FRINGE as i32;
                        self.state[self.angles[left].index] = GP3Type::FRINGE as i32;
                        self.state[self.angles[right].index] = GP3Type::FRINGE as i32;
                
                        self.ffn[self.r] = self.angles[left].index;
                        self.sfn[self.r] = self.angles[right].index;
                        self.ffn[self.angles[left].index] = self.angles[right].index;
                        self.sfn[self.angles[left].index] = self.r;
                        self.ffn[self.angles[right].index] = self.r;
                        self.sfn[self.angles[right].index] = self.angles[left].index;
                
                        nr_parts += 1;
                    }
                
                    // 查找下一个自由点
                    is_free = -1;
                    for i in 0..self.state.len() {
                        if self.state[i] == GP3Type::FREE as i32 {
                            is_free = i as i32;
                            break;
                        }
                    }
                }// end while
                /* end 补充逻辑 */
                
                
            }
    
            // 查找下一个自由点
            is_free = -1;
            for i in 0..self.state.len() {
                if self.state[i] == GP3Type::FREE as i32 {
                    is_free = i as i32;
                    break;
                }
            }
        }
    
        return true;
    }
    
    // 5.25 连接点
    pub fn connect_point(&mut self, polygons: &mut Vec<Vertices>, prev_index: usize, next_index: usize, next_next_index: usize, uvn_current: Vector2<f64>, uvn_prev: Vector2<f64>, uvn_next: Vector2<f64>) {
        if self.is_current_free {
            self.ffn[self.current_index] = prev_index;
            self.sfn[self.current_index] = next_index;
        } else {
            if (self.prev_is_ffn && self.next_is_sfn) || (self.prev_is_sfn && self.next_is_ffn) {
                self.state[self.current_index] = GP3Type::COMPLETED as i32;
            } else if self.prev_is_ffn && !self.next_is_sfn {
                self.ffn[self.current_index] = next_index;
            } else if self.next_is_ffn && !self.prev_is_sfn {
                self.ffn[self.current_index] = prev_index;
            } else if self.prev_is_sfn && !self.next_is_ffn {
                self.sfn[self.current_index] = next_index;
            } else if self.next_is_sfn && !self.prev_is_ffn {
                self.sfn[self.current_index] = prev_index;
            } else {
                // 更复杂的逻辑，根据可见性判断
                let mut found_triangle = false;
                if (prev_index != self.r) && ((self.ffn[self.current_index] == self.ffn[prev_index]) || (self.ffn[self.current_index] == self.sfn[prev_index])) {
                    found_triangle = true;
                    self.add_triangle(self.current_index, self.ffn[self.current_index], prev_index, polygons);
                    self.state[prev_index] = GP3Type::COMPLETED as i32;
                    self.state[self.ffn[self.current_index]] = GP3Type::COMPLETED as i32;
                    self.ffn[self.current_index] = next_index;
                } else if (prev_index != self.r) && ((self.sfn[self.current_index] == self.ffn[prev_index]) || (self.sfn[self.current_index] == self.sfn[prev_index])) {
                    found_triangle = true;
                    self.add_triangle(self.current_index, self.sfn[self.current_index], prev_index, polygons);
                    self.state[prev_index] = GP3Type::COMPLETED as i32;
                    self.state[self.sfn[self.current_index]] = GP3Type::COMPLETED as i32;
                    self.sfn[self.current_index] = next_index;
                } else if self.state[next_index] > GP3Type::FREE as i32 {
                    if (self.ffn[self.current_index] == self.ffn[next_index]) || (self.ffn[self.current_index] == self.sfn[next_index]) {
                        found_triangle = true;
                        self.add_triangle(self.current_index, self.ffn[self.current_index], next_index, polygons);
    
                        if self.ffn[self.current_index] == self.ffn[next_index] {
                            self.ffn[next_index] = self.current_index;
                        } else {
                            self.sfn[next_index] = self.current_index;
                        }
                        self.state[self.ffn[self.current_index]] = GP3Type::COMPLETED as i32;
                        self.ffn[self.current_index] = prev_index;
                    } else if (self.sfn[self.current_index] == self.ffn[next_index]) || (self.sfn[self.current_index] == self.sfn[next_index]) {
                        found_triangle = true;
                        self.add_triangle(self.current_index, self.sfn[self.current_index], next_index, polygons);
    
                        if self.sfn[self.current_index] == self.ffn[next_index] {
                            self.ffn[next_index] = self.current_index;
                        } else {
                            self.sfn[next_index] = self.current_index;
                        }
                        self.state[self.sfn[self.current_index]] = GP3Type::COMPLETED as i32;
                        self.sfn[self.current_index] = prev_index;
                    }
                }
    
                if found_triangle {
                    // 更新状态
                } else {
                    // 更复杂的逻辑，根据可见性判断
                    let mut prev_ffn = self.is_visible(&uvn_prev, &uvn_next, &uvn_current, &self.uvn_ffn) && self.is_visible(&uvn_prev, &self.uvn_sfn, &uvn_current, &self.uvn_ffn);
                    let mut prev_sfn = self.is_visible(&uvn_prev, &uvn_next, &uvn_current, &self.uvn_sfn) && self.is_visible(&uvn_prev, &self.uvn_ffn, &uvn_current, &self.uvn_sfn);
                    let mut next_ffn = self.is_visible(&uvn_next, &uvn_prev, &uvn_current, &self.uvn_ffn) && self.is_visible(&uvn_next, &self.uvn_sfn, &uvn_current, &self.uvn_ffn);
                    let mut next_sfn = self.is_visible(&uvn_next, &uvn_prev, &uvn_current, &self.uvn_sfn) && self.is_visible(&uvn_next, &self.uvn_ffn, &uvn_current, &self.uvn_sfn);
    
                    // 根据可见性判断连接方式
                    // ...
                }
            }
        }
    }
    
    // 5.26 封闭三角形
    pub fn close_triangle(&mut self, polygons: &mut Vec<Vertices>) {
        self.state[self.r] = GP3Type::COMPLETED as i32;
        self.add_triangle(self.angles[0].index, self.angles[1].index, self.r, polygons);
        for a_idx in 0..2 {
            if self.ffn[self.angles[a_idx].index] == self.r {
                if self.sfn[self.angles[a_idx].index] == self.angles[(a_idx + 1) % 2].index {
                    self.state[self.angles[a_idx].index] = GP3Type::COMPLETED as i32;
                } else {
                    self.ffn[self.angles[a_idx].index] = self.angles[(a_idx + 1) % 2].index;
                }
            } else if self.sfn[self.angles[a_idx].index] == self.r {
                if self.ffn[self.angles[a_idx].index] == self.angles[(a_idx + 1) % 2].index {
                    self.state[self.angles[a_idx].index] = GP3Type::COMPLETED as i32;
                } else {
                    self.sfn[self.angles[a_idx].index] = self.angles[(a_idx + 1) % 2].index;
                }
            }
        }
    }
    
    // 5.27 添加三角形
    pub fn add_triangle(&mut self, a: usize, b: usize, c: usize, polygons: &mut Vec<Vertices>) {
        // 调整 triangle.vertices 的大小为 3，初始值为 0
        self.triangle.vertices.resize(3, 0);
        if self.consistent_ordering {
            // 获取索引 a 对应的点
            let p = self.input.as_ref().unwrap().points[self.indices.as_ref().unwrap()[a]].clone();
            // 将点 p 的坐标转换为 Vector3 类型
            let pv = Vector3::new(p.point.x as f64, p.point.y as f64, p.point.z as f64);
            
            // 获取索引 b 对应的点的向量
            let pb = Vector3::new(
                self.input.as_ref().unwrap().points[self.indices.as_ref().unwrap()[b]].point.x as f64,
                self.input.as_ref().unwrap().points[self.indices.as_ref().unwrap()[b]].point.y as f64,
                self.input.as_ref().unwrap().points[self.indices.as_ref().unwrap()[b]].point.z as f64,
            );
            
            // 获取索引 c 对应的点的向量
            let pc = Vector3::new(
                self.input.as_ref().unwrap().points[self.indices.as_ref().unwrap()[c]].point.x as f64,
                self.input.as_ref().unwrap().points[self.indices.as_ref().unwrap()[c]].point.y as f64,
                self.input.as_ref().unwrap().points[self.indices.as_ref().unwrap()[c]].point.z as f64,
            );
            
            // 计算向量差
            let vec1 = pv - pb;
            let vec2 = pv - pc;
            
            // 计算叉积
            let cross_product = vec1.cross(&vec2);
            
            // 将 p.point.normal 转换为 Vector3 类型
            let normal = Vector3::new(
                p.point.normal[0] as f64,
                p.point.normal[1] as f64,
                p.point.normal[2] as f64,
            );
            
            // 计算点积
            if normal.dot(&cross_product) > 0.0 {
                // 如果点积大于 0，按 a, b, c 的顺序存储顶点索引
                self.triangle.vertices[0] = a;
                self.triangle.vertices[1] = b;
                self.triangle.vertices[2] = c;
            } else {
                // 如果点积小于等于 0，按 a, c, b 的顺序存储顶点索引
                self.triangle.vertices[0] = a;
                self.triangle.vertices[1] = c;
                self.triangle.vertices[2] = b;
            }
        } else {
            // 如果不进行一致性排序，按 a, b, c 的顺序存储顶点索引
            self.triangle.vertices[0] = a;
            self.triangle.vertices[1] = b;
            self.triangle.vertices[2] = c;
        }
        // 将当前三角形添加到多边形列表中
        polygons.push(self.triangle.clone());
    }
    
    // // 5.28 添加边缘点
    pub fn add_fringe_point(&mut self, v: usize, s: usize) {
        self.source[v] = s;
        self.part[v] = self.part[s];
        self.fringe_queue.push(v);
    }
    
    // 5.29 按角度升序排序
    pub fn nn_angle_sort_asc(a1: &NNAngle, a2: &NNAngle) -> bool {
        if a1.visible == a2.visible {
            return a1.angle < a2.angle;
        }
        a1.visible
    }
    
    // 5.30 判断点 X 是否从点 R（或原点）可见，考虑点 S1 和 S2 之间的线段
    pub fn is_visible(&self, X: &Vector2<f64>, S1: &Vector2<f64>, S2: &Vector2<f64>, R: &Vector2<f64>) -> bool {
        let a0 = S1[1] - S2[1];
        let b0 = S2[0] - S1[0];
        let c0 = S1[0] * S2[1] - S2[0] * S1[1];
        let mut a1 = -X[1];
        let mut b1 = X[0];
        let mut c1 = 0.0;
        if *R != Vector2::zeros() {
            a1 += R[1];
            b1 -= R[0];
            c1 = R[0] * X[1] - X[0] * R[1];
        }
        let div = a0 * b1 - b0 * a1;
        let x = (b0 * c1 - b1 * c0) / div;
        let y = (a1 * c0 - a0 * c1) / div;
    
        let mut intersection_outside_XR = false;
        if *R == Vector2::zeros() {
            if X[0] > 0.0 {
                intersection_outside_XR = (x <= 0.0) || (x >= X[0]);
            } else if X[0] < 0.0 {
                intersection_outside_XR = (x >= 0.0) || (x <= X[0]);
            } else if X[1] > 0.0 {
                intersection_outside_XR = (y <= 0.0) || (y >= X[1]);
            } else if X[1] < 0.0 {
                intersection_outside_XR = (y >= 0.0) || (y <= X[1]);
            } else {
                intersection_outside_XR = true;
            }
        } else {
            if X[0] > R[0] {
                intersection_outside_XR = (x <= R[0]) || (x >= X[0]);
            } else if X[0] < R[0] {
                intersection_outside_XR = (x >= R[0]) || (x <= X[0]);
            } else if X[1] > R[1] {
                intersection_outside_XR = (y <= R[1]) || (y >= X[1]);
            } else if X[1] < R[1] {
                intersection_outside_XR = (y >= R[1]) || (y <= X[1]);
            } else {
                intersection_outside_XR = true;
            }
        }
        if intersection_outside_XR {
            return true;
        }
        if S1[0] > S2[0] {
            return (x <= S2[0]) || (x >= S1[0]);
        }
        if S1[0] < S2[0] {
            return (x >= S2[0]) || (x <= S1[0]);
        }
        if S1[1] > S2[1] {
            return (y <= S2[1]) || (y >= S1[1]);
        }
        if S1[1] < S2[1] {
            return (y >= S2[1]) || (y <= S1[1]);
        }
        false
    }
    
}

/* end 实现 */

#[cfg(test)]
mod tests5 {
    use super::*;

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
}
