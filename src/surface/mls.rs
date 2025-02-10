#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]

//! 最小二乘搜索法

// 标准库
use std::collections::{ HashMap, BTreeMap };
use std::sync::Arc;
use std::marker::PhantomData;
use std::cell::RefCell;
use std::borrow::{BorrowMut, Borrow};

// 随机数
use rand::Rng;
use rand_distr::Uniform;

// 元编程
use paste::paste;

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
    PointXYZRGB, PointXYZRGBNormal, Normal,
};

/* start 结构体 */
// 1. 定义 MLSResult 结构体，用于存储 MLS 拟合的结果
#[derive(Default, Clone)]
pub struct MLSResult {
    // 查询点
    pub query_point: Vector3<f64>, 
    // 邻域的均值
    pub mean: Vector3<f64>,         
    // 查询点的平面法向量
    pub plane_normal: Vector3<f64>, 
    // u 方向的轴
    pub u_axis: Vector3<f64>,     
    // v 方向的轴 
    pub v_axis: Vector3<f64>,    
    // 多项式系数   
    pub c_vec: Vec<f64>,           
    // 邻居数量 
    pub num_neighbors: usize,        
    // 曲率
    pub curvature: f32,              
    // 多项式的阶数
    pub order: usize,                
    // 结果是否有效
    pub valid: bool,                 
}

// 2. 定义 MovingLeastSquares 结构体
pub struct MovingLeastSquares<PointInT, PointOutT> {
    // 输入点云
    pub input: Arc<Vec<PointInT>>,
    // 输出点云
    pub output: Vec<PointOutT>,
    // 邻居搜索半径
    pub search_radius: f64,
    // 多项式阶数
    pub order: usize,
    // 是否计算法线
    pub compute_normals: bool,
    // 存储 MLS 结果
    pub mls_results: Vec<MLSResult>,
    // 不同的点云，用于DISTINCT_CLOUD上采样方法
    pub distinct_cloud: Arc<Vec<PointInT>>,
    // 上采样方法
    pub upsample_method: UpsamplingMethod,
    // 上采样半径，仅用于SAMPLE_LOCAL_PLANE上采样
    pub upsampling_radius: f64,
    // 上采样步长，仅用于SAMPLE_LOCAL_PLANE上采样
    pub upsampling_step: f64,
    // 搜索半径内期望的点数，仅用于RANDOM_UNIFORM_DENSITY上采样
    pub desired_num_points_in_radius: usize,
    // 是否缓存MLS结果
    pub cache_mls_results: bool,
    // 投影方法
    pub projection_method: ProjectionMethod,
    // 最大线程数
    pub threads: usize,
    // 体素大小，仅用于VOXEL_GRID_DILATION上采样方法
    pub voxel_size: f32,
    // 体素网格的膨胀迭代次数，仅用于VOXEL_GRID_DILATION上采样方法
    pub dilation_iteration_num: usize,
    // 收集输出中每个点对应的输入点索引
    pub corresponding_input_indices: Vec<usize>,
    // 随机数生成器
    pub rng: rand::rngs::ThreadRng,
    // 随机数生成器使用的均匀分布，仅用于RANDOM_UNIFORM_DENSITY上采样
    pub rng_uniform_distribution: Option<rand_distr::Uniform<f64>>,
}

// 3. 定义多项式偏导数结构体
#[derive(Default)]
pub struct PolynomialPartialDerivative {
    pub z: f64,
    pub z_u: f64,
    pub z_v: f64,
    pub z_uu: f64,
    pub z_vv: f64,
    pub z_uv: f64,
}

// 4. MLS投影结果结构体
#[derive(Default)]
pub struct MLSProjectionResults {
    // The u-coordinate of the projected point in local MLS frame.
    pub u: f64,     
    // The v-coordinate of the projected point in local MLS frame.    
    pub v: f64,               
    // The projected point.
    pub point: Vector3<f64>,  
    // The projected point's normal.
    pub normal: Vector3<f64>, 
}

// 5. 定义 MLSVoxelGrid 结构体
#[derive(Default, Clone)]
pub struct MLSVoxelGrid<PointT> {
    // 存储体素网格的映射，键为一维索引，值为 Leaf 结构体
    pub voxel_grid: BTreeMap<u64, Leaf>,
    // 体素网格的最小边界
    pub bounding_min: Vector4<f32>,
    // 体素网格的最大边界
    pub bounding_max: Vector4<f32>,
    // 数据的大小
    pub data_size: u64,
    // 体素的大小
    pub voxel_size: f32,
    // 点类型
    _marker: PhantomData<PointT>,
}

// 6. 定义 Leaf 结构体，表示体素网格中的叶子节点
#[derive(Default, Clone)]
pub struct Leaf {
    // 表示该叶子节点是否有效
    pub valid: bool,
}

/* end 结构体 */

/* start 枚举 */

// 1. 投影方法
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ProjectionMethod{
    // 投影到MLS平面。
    NONE,      
    // 沿MLS平面的法线投影到多项式表面。
    SIMPLE,    
    // 投影到多项式表面上最近的点。
    ORTHOGONAL,
}

// 2. 上采样方法
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum UpsamplingMethod{
    // 不进行上采样，仅将输入点投影到它们自己的MLS表面。
    NONE,                   
    // 将不同云的点投影到MLS表面。
    DISTINCT_CLOUD,        
    // 每个输入点的局部平面将使用上采样半径和上采样步长参数以圆形方式进行采样。
    SAMPLE_LOCAL_PLANE,     
    // 每个输入点的局部平面将使用均匀随机分布进行采样，以确保点的密度在整个云中保持恒定 - 由期望半径内的点数参数给出。
    RANDOM_UNIFORM_DENSITY,
    // 输入云将被插入到一个体素网格中，体素大小由体素大小参数给出；该体素网格将膨胀指定次数，结果点将投影到输入云中最近点的MLS表面；结果是一个填充孔洞且点密度恒定的点云。
    VOXEL_GRID_DILATION     
}
/* end 枚举 */

/* start 实现 */
// 在 MLSResult 结构体实现外部定义一个辅助函数
fn default_weight_func(sq_dist: f64, sq_mls_radius: f64) -> f64 {
    use std::f64::consts::E;
    E.powf(-sq_dist / sq_mls_radius)
}

// 1. MLSResult 结构体，用于存储 MLS 拟合的结果
impl MLSResult {
    // 1.1 构造函数
    pub fn new(
        query_point: Vector3<f64>,
        mean: Vector3<f64>,
        plane_normal: Vector3<f64>,
        u: Vector3<f64>,
        v: Vector3<f64>,
        c_vec: Vec<f64>,
        num_neighbors: usize,
        curvature: f32,
        order: usize,
    ) -> Self {
        Self {
            query_point,
            mean,
            plane_normal,
            u_axis: u,
            v_axis: v,
            c_vec,
            num_neighbors,
            curvature,
            order,
            valid: true,
        }
    }

    // 1.2 计算给定点在 MLS 坐标系中的 3D 位置
    pub fn get_mls_coordinates(&self, pt: &Vector3<f64>) -> (f64, f64, f64) {
        let delta = pt - self.mean;
        let u = delta.dot(&self.u_axis);
        let v = delta.dot(&self.v_axis);
        let w = delta.dot(&self.plane_normal);
        (u, v, w)
    }

    // 1.3 计算多项式值
    pub fn get_polynomial_value(&self, u: f64, v: f64) -> f64 {
        let mut result = 0.0;
        let mut j = 0;
        let mut u_pow = 1.0;

        for ui in 0..=self.order {
            let mut v_pow = 1.0;
            for vi in 0..=self.order - ui {
                result += self.c_vec[j] * u_pow * v_pow;
                v_pow *= v;
                j += 1;
            }
            u_pow *= u;
        }

        result
    }

    // 1.4 计算多项式的偏导数
    pub fn get_polynomial_partial_derivative(&self, u: f64, v: f64) -> PolynomialPartialDerivative {
        let mut d = PolynomialPartialDerivative::default();
        let mut u_pow = vec![1.0; self.order + 2];
        let mut v_pow = vec![1.0; self.order + 2];
        let mut j = 0;

        for ui in 0..=self.order {
            for vi in 0..=self.order - ui {
                d.z += u_pow[ui] * v_pow[vi] * self.c_vec[j];

                if ui >= 1 {
                    d.z_u += self.c_vec[j] * ui as f64 * u_pow[ui - 1] * v_pow[vi];
                }

                if vi >= 1 {
                    d.z_v += self.c_vec[j] * vi as f64 * u_pow[ui] * v_pow[vi - 1];
                }

                if ui >= 1 && vi >= 1 {
                    d.z_uv += self.c_vec[j] * ui as f64 * u_pow[ui - 1] * vi as f64 * v_pow[vi - 1];
                }

                if ui >= 2 {
                    d.z_uu += self.c_vec[j] * ui as f64 * (ui - 1) as f64 * u_pow[ui - 2] * v_pow[vi];
                }

                if vi >= 2 {
                    d.z_vv += self.c_vec[j] * vi as f64 * (vi - 1) as f64 * u_pow[ui] * v_pow[vi - 2];
                }

                if ui == 0 {
                    v_pow[vi + 1] = v_pow[vi] * v;
                }

                j += 1;
            }
            u_pow[ui + 1] = u_pow[ui] * u;
        }
        // 返回值
        d
    }

    // 1.5 计算主曲率
    pub fn calculate_principal_curvatures(&self, u: f64, v: f64) -> Vector2<f32> {
        let mut k = Vector2::new(1e-5 as f32, 1e-5 as f32);

        // 检查多项式阶数和系数的有效性
        if self.order > 1 && self.c_vec.len() >= (self.order + 1) * (self.order + 2) / 2 {
            let d = self.get_polynomial_partial_derivative(u, v);
            let z = 1.0 + d.z_u * d.z_u + d.z_v * d.z_v;
            let zlen = z.sqrt();
            let k_value = (d.z_uu * d.z_vv - d.z_uv * d.z_uv) / (z * z);
            let h = ((1.0 + d.z_v * d.z_v) * d.z_uu - 2.0 * d.z_u * d.z_v * d.z_uv + (1.0 + d.z_u * d.z_u) * d.z_vv) / (2.0 * zlen * zlen * zlen);
            let disc2 = h * h - k_value;
            assert!(disc2 >= 0.0);
            let disc = disc2.sqrt();
            k[0] = (h + disc) as f32;
            k[1] = (h - disc) as f32;

            if k[0].abs() > k[1].abs() {
                // 交换 k[0] 和 k[1]
                k.swap((0, 0), (1,1));
            }
        } else {
            eprintln!("没有多项式拟合数据，无法计算主曲率！");
        }
        // 返回值
        k
    }
    
    // 1.6 计算 MLS 权重
    fn compute_mls_weight(&self, sq_dist: f64, sq_mls_radius: f64) -> f64 {
        use std::f64::consts::E;
        E.powf(-sq_dist / sq_mls_radius)
    }
    
    // 1.7 将点正交投影到多项式表面
    pub fn project_point_orthogonal_to_polynomial_surface(&self, u: f64, v: f64, w: f64) -> MLSProjectionResults {
        let mut gu = u;
        let mut gv = v;
        let mut gw = 0.0;

        let mut result = MLSProjectionResults::default();
        result.normal = self.plane_normal;

        if self.order > 1 && self.c_vec.len() >= (self.order + 1) * (self.order + 2) / 2 && self.c_vec[0].is_finite() {
            let mut d = self.get_polynomial_partial_derivative(gu, gv);
            gw = d.z;
            let mut err_total;
            let dist1 = (gw - w).abs();
            let mut dist2;

            loop {
                let e1 = (gu - u) + d.z_u * gw - d.z_u * w;
                let e2 = (gv - v) + d.z_v * gw - d.z_v * w;

                let f1u = 1.0 + d.z_uu * gw + d.z_u * d.z_u - d.z_uu * w;
                let f1v = d.z_uv * gw + d.z_u * d.z_v - d.z_uv * w;

                let f2u = d.z_uv * gw + d.z_v * d.z_u - d.z_uv * w;
                let f2v = 1.0 + d.z_vv * gw + d.z_v * d.z_v - d.z_vv * w;

                let j = Matrix2::new(f1u, f1v, f2u, f2v);
                let err = Vector2::new(e1, e2);
                let update = j.try_inverse().unwrap() * err;
                gu -= update[0];
                gv -= update[1];

                d = self.get_polynomial_partial_derivative(gu, gv);
                gw = d.z;
                dist2 = ((gu - u) * (gu - u) + (gv - v) * (gv - v) + (gw - w) * (gw - w)).sqrt();

                err_total = (e1 * e1 + e2 * e2).sqrt();

                if err_total <= 1e-8 || dist2 >= dist1 {
                    break;
                }
            }

            if dist2 > dist1 {
                gu = u;
                gv = v;
                d = self.get_polynomial_partial_derivative(u, v);
                gw = d.z;
            }

            result.u = gu;
            result.v = gv;
            result.normal -= d.z_u * self.u_axis + d.z_v * self.v_axis;
            result.normal.normalize_mut();
        }

        result.point = self.mean + gu * self.u_axis + gv * self.v_axis + gw * self.plane_normal;

        result
    }
    
    // 1.8 将点投影到 MLS 平面
    pub fn project_point_to_mls_plane(&self, u: f64, v: f64) -> MLSProjectionResults {
        let mut result = MLSProjectionResults::default();
        result.u = u;
        result.v = v;
        result.normal = self.plane_normal;
        result.point = self.mean + u * self.u_axis + v * self.v_axis;

        result
    }

    // 1.9 将点沿 MLS 平面法线投影到多项式表面
    pub fn project_point_simple_to_polynomial_surface(&self, u: f64, v: f64) -> MLSProjectionResults {
        let mut result = MLSProjectionResults::default();
        let mut w = 0.0;

        result.u = u;
        result.v = v;
        result.normal = self.plane_normal;

        if self.order > 1 && self.c_vec.len() >= (self.order + 1) * (self.order + 2) / 2 && self.c_vec[0].is_finite() {
            let d = self.get_polynomial_partial_derivative(u, v);
            w = d.z;
            result.normal -= d.z_u * self.u_axis + d.z_v * self.v_axis;
            result.normal.normalize_mut();
        }

        result.point = self.mean + u * self.u_axis + v * self.v_axis + w * self.plane_normal;

        result
    }
    
    // 1.10 使用指定方法投影点
    pub fn project_point(&self, pt: &Vector3<f64>, method: ProjectionMethod, required_neighbors: usize) -> MLSProjectionResults {
        let (u, v, w) = self.get_mls_coordinates(pt);

        let mut proj = MLSProjectionResults::default();
        if self.order > 1 && self.num_neighbors >= required_neighbors && self.c_vec[0].is_finite() && method != ProjectionMethod::NONE {
            if method == ProjectionMethod::ORTHOGONAL {
                proj = self.project_point_orthogonal_to_polynomial_surface(u, v, w);
            } else {
                proj = self.project_point_simple_to_polynomial_surface(u, v);
            }
        } else {
            proj = self.project_point_to_mls_plane(u, v);
        }
        // 返回值
        proj
    }
    
    // 1.11 投影用于生成 MLS 表面的查询点
    pub fn project_query_point(&self, method: ProjectionMethod, required_neighbors: usize) -> MLSProjectionResults {
        let mut proj = MLSProjectionResults::default();
        if self.order > 1 && self.num_neighbors >= required_neighbors && self.c_vec[0].is_finite() && method != ProjectionMethod::NONE {
            if method == ProjectionMethod::ORTHOGONAL {
                let (u, v, w) = self.get_mls_coordinates(&self.query_point);
                proj = self.project_point_orthogonal_to_polynomial_surface(u, v, w);
            } else {
                proj.point = self.mean + self.c_vec[0] * self.plane_normal;
                proj.normal = self.plane_normal - self.c_vec[self.order + 1] * self.u_axis - self.c_vec[1] * self.v_axis;
                proj.normal.normalize_mut();
            }
        } else {
            proj.normal = self.plane_normal;
            proj.point = self.mean;
        }
        // 返回值
        proj
    }
    
    // 1.12 计算 MLS 表面
    pub fn compute_mls_surface<PointT>(
        &mut self,
        cloud: &[PointT],
        index: usize,
        nn_indices: &[usize],
        search_radius: f64,
        polynomial_order: usize,
        weight_func: Option<Box<dyn Fn(f64) -> f64>>,
    ) where
        PointT: Copy + Into<Vector3<f64>>,
        DefaultAllocator: Allocator<Const<3>, Const<3>> + Allocator<Const<3>, Const<3>>,
    {
        // 计算平面系数
        let mut covariance_matrix = DMatrix::<f64>::zeros(3, 3);
        let mut xyz_centroid = Vector3::zeros();
    
        // 估计 XYZ 质心
        for &i in nn_indices {
            let point: Vector3<f64> = cloud[i].into();
            xyz_centroid += point;
        }
        xyz_centroid /= nn_indices.len() as f64;
    
        // 计算 3x3 协方差矩阵
        for &i in nn_indices {
            let point: Vector3<f64> = cloud[i].into();
            let diff = point - xyz_centroid;
            covariance_matrix += diff * diff.transpose();
        }
        covariance_matrix /= nn_indices.len() as f64;
    
        let eigen_result = covariance_matrix.clone().symmetric_eigen();
        let eigen_value = eigen_result.eigenvalues[0];
        let eigen_vector = eigen_result.eigenvectors.column(0);
        let mut model_coefficients = Vector3::zeros();
        model_coefficients = eigen_vector.fixed_view::<3, 1>(0, 0).into_owned();
        let d = -model_coefficients.dot(&xyz_centroid);
    
        self.query_point = cloud[index].into();
    
        if !eigen_vector[0].is_finite() || !eigen_vector[1].is_finite() || !eigen_vector[2].is_finite() {
            // 无效的平面系数，可能是输入云是非密集的（包含无效点）
            // 保留输入点并在此处停止
            self.valid = false;
            self.mean = self.query_point;
            return;
        }
    
        // 投影查询点
        self.valid = true;
        let distance = self.query_point.dot(&model_coefficients) + d;
        self.mean = self.query_point - distance * model_coefficients;
    
        self.curvature = covariance_matrix.trace() as f32;
        // 计算曲率表面变化
        if self.curvature != 0.0 {
            self.curvature = (eigen_value / self.curvature as f64).abs() as f32;
        }
    
        // 获取平面法线的副本以便于访问
        self.plane_normal = model_coefficients;
    
        // 局部坐标系（Darboux 框架）
        let mut temp_vec = Vector3::new(1.0, 0.0, 0.0);
        if self.plane_normal.cross(&temp_vec).norm() < 1e-8 {
            temp_vec = Vector3::new(0.0, 1.0, 0.0);
        }
        self.v_axis = self.plane_normal.cross(&temp_vec).normalize();
        self.u_axis = self.plane_normal.cross(&self.v_axis);
    
        // 执行多项式拟合以更新点和法线
        self.num_neighbors = nn_indices.len();
        self.order = polynomial_order;
        if self.order > 1 {
            let nr_coeff = (self.order + 1) * (self.order + 2) / 2;
    
            if self.num_neighbors >= nr_coeff {
                let weight_func = weight_func.unwrap_or_else(|| {
                    let sq_mls_radius = search_radius * search_radius;
                    Box::new(move |sq_dist| default_weight_func(sq_dist, sq_mls_radius))
                });
    
                // 分配矩阵和向量以保存多项式拟合所需的数据
                let mut weight_vec = DVector::<f64>::zeros(self.num_neighbors);
                let mut P = DMatrix::<f64>::zeros(nr_coeff, self.num_neighbors);
                let mut f_vec = DVector::<f64>::zeros(self.num_neighbors);
                let mut P_weight_Pt = DMatrix::<f64>::zeros(nr_coeff, nr_coeff);
    
                // 更新邻域，因为点已投影，并计算相对位置
                // 注意：仅更新权重的距离以提高速度
                let mut de_meaned = vec![Vector3::zeros(); self.num_neighbors];
                for (ni, &idx) in nn_indices.iter().enumerate() {
                    let point: Vector3<f64> = cloud[idx].into();
                    de_meaned[ni] = point - self.mean;
                    weight_vec[ni] = weight_func(de_meaned[ni].norm_squared());
                }
    
                // 遍历邻居，将它们转换到局部坐标系中，保存高度和多项式项的求值结果
                for (ni, &idx) in nn_indices.iter().enumerate() {
                    // 转换坐标
                    let u_coord = de_meaned[ni].dot(&self.u_axis);
                    let v_coord = de_meaned[ni].dot(&self.v_axis);
                    f_vec[ni] = de_meaned[ni].dot(&self.plane_normal);
    
                    // 计算当前点处多项式的项
                    let mut j = 0;
                    let mut u_pow = 1.0;
                    for ui in 0..=self.order {
                        let mut v_pow = 1.0;
                        for vi in 0..=self.order - ui {
                            P[(j, ni)] = u_pow * v_pow;
                            v_pow *= v_coord;
                            j += 1;
                        }
                        u_pow *= u_coord;
                    }
                }
    
                // 计算系数
                let P_weight = P.clone() * DMatrix::from_diagonal(&weight_vec);
                P_weight_Pt = P_weight.clone() * P.transpose();
                self.c_vec = (P_weight * f_vec).data.as_vec().clone();
                let chol = P_weight_Pt.cholesky().expect("Cholesky decomposition failed");
                self.c_vec = chol.solve(&DVector::from_vec(self.c_vec.clone())).data.as_vec().clone();
            }
        }
    }
    
}

// 2. MovingLeastSquares的实现
impl<PointInT, PointOutT> MovingLeastSquares<PointInT, PointOutT> {
    // 2.1 构造函数
    pub fn new() -> Self {
        Self {
            input: Arc::new(Vec::new()),
            output: Vec::new(),
            search_radius: 0.0,
            order: 2,
            compute_normals: false,
            mls_results: Vec::new(),
            distinct_cloud: Arc::new(Vec::new()),
            upsample_method: UpsamplingMethod::NONE,
            upsampling_radius: 0.0,
            upsampling_step: 0.0,
            desired_num_points_in_radius: 0,
            cache_mls_results: true,
            projection_method: ProjectionMethod::SIMPLE,
            threads: 1,
            voxel_size: 1.0,
            dilation_iteration_num: 0,
            corresponding_input_indices: vec![],
            rng: rand::rng(),
            rng_uniform_distribution: None,
        }
    }

    // 2.2 设置输入点云
    pub fn set_input_cloud(&mut self, cloud: Arc<Vec<PointInT>>) {
        self.input = cloud;
    }

    // 2.3 设置搜索半径
    pub fn set_search_radius(&mut self, radius: f64) {
        self.search_radius = radius;
    }

    // 2.4 设置多项式阶数
    pub fn set_polynomial_order(&mut self, order: usize) {
        self.order = order;
    }

    // 2.5 设置是否计算法线
    pub fn set_compute_normals(&mut self, compute: bool) {
        self.compute_normals = compute;
    }
    
    // 2.6 设置不同的点云，用于DISTINCT_CLOUD上采样方法
    pub fn set_distinct_cloud(&mut self, cloud: Arc<Vec<PointInT>>) {
        self.distinct_cloud = cloud;
    }
    
    // 2.7 设置上采样方法
    pub fn set_upsampling_method(&mut self, method: UpsamplingMethod) {
        self.upsample_method = method;
    }
    
    // 2.8 设置上采样半径，仅用于SAMPLE_LOCAL_PLANE上采样
    pub fn set_upsampling_radius(&mut self, radius: f64) {
        self.upsampling_radius = radius;
    }
    
    // 2.9 设置上采样步长，仅用于SAMPLE_LOCAL_PLANE上采样
    pub fn set_upsampling_step_size(&mut self, step_size: f64) {
        self.upsampling_step = step_size;
    }
    
    // 2.10 设置搜索半径内期望的点数，仅用于RANDOM_UNIFORM_DENSITY上采样
    pub fn set_point_density(&mut self, num_points: usize) {
        self.desired_num_points_in_radius = num_points;
    }
    
    // 2.11 设置是否缓存MLS结果
    pub fn set_cache_mls_results(&mut self, cache: bool) {
        self.cache_mls_results = cache;
    }
    
    // 2.12 设置投影方法
    pub fn set_projection_method(&mut self, method: ProjectionMethod) {
        self.projection_method = method;
    }
    
    // 2.13 设置最大线程数
    pub fn set_number_of_threads(&mut self, threads: usize) {
        self.threads = threads;
    }
    
    // 2.14 设置体素大小，仅用于VOXEL_GRID_DILATION上采样方法
    pub fn set_dilation_voxel_size(&mut self, size: f32) {
        self.voxel_size = size;
    }
    
    // 2.15 设置体素网格的膨胀迭代次数，仅用于VOXEL_GRID_DILATION上采样方法
    pub fn set_dilation_iterations(&mut self, iterations: usize) {
        self.dilation_iteration_num = iterations;
    }
    
}

// 先实现特定类型
impl MovingLeastSquares<PointXYZRGBNormal, PointXYZRGBNormal> {
    // 2.16 处理点云的主要方法
    pub fn process(&mut self) {
        // 重置或初始化对应输入索引
        self.corresponding_input_indices.clear();

        // 初始化法线向量
        let mut normals = vec![];
        if self.compute_normals {
            normals = vec![Normal::default(); self.input.len()];
        }

        // 清空输出点云
        self.output.clear();

        // 检查搜索半径和高斯参数是否有效
        if self.search_radius <= 0.0 {
            eprintln!("[MovingLeastSquares::process] 无效的搜索半径: {}", self.search_radius);
            return;
        }

        // 检查DISTINCT_CLOUD上采样方法是否设置了不同的点云
        if self.upsample_method == UpsamplingMethod::DISTINCT_CLOUD && self.distinct_cloud.is_empty() {
            eprintln!("[MovingLeastSquares::process] 上采样方法设置为DISTINCT_CLOUD，但未指定不同的点云。");
            return;
        }

        // 初始化随机数生成器
        if self.upsample_method == UpsamplingMethod::RANDOM_UNIFORM_DENSITY {
            let tmp = self.search_radius / 2.0;
            self.rng_uniform_distribution = Some(Uniform::new(-tmp, tmp).expect("REASON"));
        }

        // 确保在VOXEL_GRID_DILATION或DISTINCT_CLOUD上采样方法下缓存MLS结果
        if self.upsample_method == UpsamplingMethod::VOXEL_GRID_DILATION || self.upsample_method == UpsamplingMethod::DISTINCT_CLOUD {
            if !self.cache_mls_results {
                eprintln!("使用VOXEL_GRID_DILATION或DISTINCT_CLOUD上采样方法时，强制缓存MLS结果。");
            }
            self.cache_mls_results = true;
        }

        // 调整MLS结果向量的大小
        if self.cache_mls_results {
            self.mls_results.resize(self.input.len(), MLSResult::default());
        } else {
            self.mls_results.resize(1, MLSResult::default());
        }

        // 执行实际的表面重建
        self.perform_processing(&mut normals);

        // 如果需要计算法线，将法线信息复制到输出点云
        if self.compute_normals {
            for (i, point) in self.output.iter_mut().enumerate() {
                // self.copy_normal_info(point, &normals[i]);
                point.normal[0] = normals[i].normal[0];
                point.normal[1] = normals[i].normal[1];
                point.normal[2] = normals[i].normal[2];
            }
        }

        // 设置输出点云的宽度和高度
        // self.output.height = 1;
        // self.output.width = self.output.len();
    }
    
    // 2.17 执行表面重建的具体处理
    fn perform_processing(&mut self, normals: &mut Vec<Normal>) {
        // 计算多项式系数的数量
        let nr_coeff = (self.order + 1) * (self.order + 2) / 2;
    
        // 单线程处理每个点
        for cp in 0..self.input.len() {
            let nn_indices = self.find_neighbors(cp, self.search_radius);
            if nn_indices.len() >= 3 {
                let mut mls_result = MLSResult::default();
                mls_result.compute_mls_surface(&self.input, cp, &nn_indices, self.search_radius, self.order, None);
    
                let mut projected_points = vec![];
                let mut projected_points_normals = vec![];
                let mut corresponding_input_indices = vec![];
    
                self.compute_mls_point_normal(cp, &nn_indices, &mut projected_points, &mut projected_points_normals, &mut corresponding_input_indices, &mut mls_result);
    
                self.output.extend(projected_points);
                if self.compute_normals {
                    normals.extend(projected_points_normals);
                }
                self.corresponding_input_indices.extend(corresponding_input_indices);
                if self.cache_mls_results {
                    self.mls_results[cp] = mls_result;
                }
            } else {
                if self.cache_mls_results {
                    self.mls_results[cp] = MLSResult::default();
                }
            }
        }
    
        // 执行上采样
        self.perform_upsampling(normals);
    }
    
    // 2.18 执行上采样操作
    fn perform_upsampling(&mut self, normals: &mut Vec<Normal>) {
        let input = Arc::clone(&self.input);
        let distinct_cloud = Arc::clone(&self.distinct_cloud);
        let mls_results = Arc::new(self.mls_results.clone());
        let mut output = self.output.clone();
        let mut corresponding_input_indices = self.corresponding_input_indices.clone();
    
        match self.upsample_method {
            UpsamplingMethod::DISTINCT_CLOUD => {
                // 修正：明确类型注解
                <Vec<usize> as std::borrow::BorrowMut<[usize]>>::borrow_mut(&mut self.corresponding_input_indices);
                for dp_i in 0..distinct_cloud.len() {
                    let point = &distinct_cloud[dp_i];
                    if !Self::is_point_finite(point) {
                        continue;
                    }
                    let nn_indices = self.find_neighbors_cloud(point, 1);
                    if let Some(input_index) = nn_indices.first() {
                        let curvature = mls_results[*input_index].curvature;
                        if mls_results[*input_index].valid {
                            let add_point = Self::point_to_vector3d(*point);
                            let proj = mls_results[*input_index].project_point(&add_point, self.projection_method, 5 * (self.order + 1) * (self.order + 2) / 2);
                            self.add_projected_point_normal(
                                *input_index,
                                &proj.point,
                                &proj.normal,
                                curvature,
                                // &mut self.output,
                                &mut output,
                                normals,
                                &mut corresponding_input_indices,
                            );
                        }
                    }
                }
            }
            UpsamplingMethod::VOXEL_GRID_DILATION => {
                // 修正：明确类型注解
                <Vec<usize> as std::borrow::BorrowMut<[usize]>>::borrow_mut(&mut self.corresponding_input_indices);
                let mut voxel_grid = MLSVoxelGrid::<PointXYZRGBNormal>::new(&input, &(0..input.len()).collect(), self.voxel_size, self.dilation_iteration_num as i32);
                for _ in 0..self.dilation_iteration_num {
                    voxel_grid.dilate();
                }
                for (index_1d, _) in voxel_grid.voxel_grid.iter() {
                    let pos = MLSVoxelGrid::<PointXYZRGBNormal>::get_position(*index_1d, voxel_grid.data_size, &voxel_grid.bounding_min, voxel_grid.voxel_size);
                    let mut p = PointXYZRGBNormal::default();
                    p.x = pos[0];
                    p.y = pos[1];
                    p.z = pos[2];
                    let nn_indices = self.find_neighbors_cloud(&p, 1);
                    if let Some(input_index) = nn_indices.first() {
                        let curvature = mls_results[*input_index].curvature;
                        if mls_results[*input_index].valid {
                            let add_point = Self::point_to_vector3d(p);
                            let proj = mls_results[*input_index].project_point(&add_point, self.projection_method, 5 * (self.order + 1) * (self.order + 2) / 2);
                            self.add_projected_point_normal(
                                *input_index,
                                &proj.point,
                                &proj.normal,
                                curvature,
                                // &mut self.output,
                                &mut output,
                                normals,
                                &mut corresponding_input_indices,
                            );
                        }
                    }
                }
            }
            _ => {}
        }
    }
    
    // 2.19 查找给定点的邻居
    fn find_neighbors(&self, index: usize, radius: f64) -> Vec<usize> {
        let mut neighbors = vec![];
        let query_point = Self::point_to_vector3d(self.input[index]);
        for (i, point) in self.input.iter().enumerate() {
            let dist = (Self::point_to_vector3d(*point) - query_point).norm();
            if dist <= radius {
                neighbors.push(i);
            }
        }
        neighbors
    }
    
    // 2.20 在不同点云中查找给定点的邻居
    fn find_neighbors_cloud(&mut self, point: &PointXYZRGBNormal, k: usize) -> Vec<usize> {
        let mut neighbors = vec![];
        let mut distances = vec![];
        let query_point = Self::point_to_vector3d(*point);
        for (i, p) in self.input.iter().enumerate() {
            let dist = (Self::point_to_vector3d(*p) - query_point).norm();
            distances.push((i, dist));
        }
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for i in 0..k.min(distances.len()) {
            neighbors.push(distances[i].0);
        }
        neighbors
    }
    
    // // 2.21 计算MLS点的法线
    fn compute_mls_point_normal(
        &mut self,
        index: usize,
        nn_indices: &[usize],
        projected_points: &mut Vec<PointXYZRGBNormal>,
        projected_points_normals: &mut Vec<Normal>,
        corresponding_input_indices: &mut Vec<usize>,
        mls_result: &mut MLSResult,
    ) {
        // 计算MLS表面，注意：这里将compute_mls_surface方法添加到了MLSResult结构体中
        mls_result.compute_mls_surface(
            &self.input,
            index,
            nn_indices,
            self.search_radius,
            self.order,
            None,
        );
    
        match self.upsample_method {
            // 不进行上采样，仅将输入点投影到它们自己的MLS表面
            UpsamplingMethod::NONE => {
                // 使用指定方法投影查询点
                let proj = mls_result.project_query_point(self.projection_method, (self.order + 1) * (self.order + 2) / 2);
                // 添加投影点及其法线
                self.add_projected_point_normal(
                    index,
                    &proj.point,
                    &proj.normal,
                    mls_result.curvature,
                    projected_points,
                    projected_points_normals,
                    corresponding_input_indices,
                );
            }
            // 每个输入点的局部平面将使用上采样半径和上采样步长参数以圆形方式进行采样
            UpsamplingMethod::SAMPLE_LOCAL_PLANE => {
                let mut u_disp = -self.upsampling_radius as f32;
                while u_disp <= self.upsampling_radius as f32 {
                    let mut v_disp = -self.upsampling_radius as f32;
                    while v_disp <= self.upsampling_radius as f32 {
                        // 检查点是否在圆形采样区域内
                        if u_disp * u_disp + v_disp * v_disp < (self.upsampling_radius * self.upsampling_radius) as f32 {
                            // 将点沿MLS平面法线投影到多项式表面
                            let proj = mls_result.project_point_simple_to_polynomial_surface(u_disp as f64, v_disp as f64);
                            // 添加投影点及其法线
                            self.add_projected_point_normal(
                                index,
                                &proj.point,
                                &proj.normal,
                                mls_result.curvature,
                                projected_points,
                                projected_points_normals,
                                corresponding_input_indices,
                            );
                        }
                        v_disp += self.upsampling_step as f32;
                    }
                    u_disp += self.upsampling_step as f32;
                }
            }
            // 每个输入点的局部平面将使用均匀随机分布进行采样，以确保点的密度在整个云中保持恒定
            UpsamplingMethod::RANDOM_UNIFORM_DENSITY => {
                // 计算需要添加的点数
                let num_points_to_add = ((self.desired_num_points_in_radius as f64 / 2.0 / nn_indices.len() as f64).floor()) as usize;
                if num_points_to_add <= 0 {
                    // 密度足够，仅添加查询点
                    let proj = mls_result.project_query_point(self.projection_method, (self.order + 1) * (self.order + 2) / 2);
                    self.add_projected_point_normal(
                        index,
                        &proj.point,
                        &proj.normal,
                        mls_result.curvature,
                        projected_points,
                        projected_points_normals,
                        corresponding_input_indices,
                    );
                } else {
                    // 采样局部平面
                    let mut num_added = 0;
                    while num_added < num_points_to_add {
                        // 生成随机偏移量
                        let u = self.rng.sample(self.rng_uniform_distribution.as_ref().unwrap());
                        let v = self.rng.sample(self.rng_uniform_distribution.as_ref().unwrap());
                        // 检查偏移量是否在圆形区域内
                        if u * u + v * v > self.search_radius * self.search_radius / 4.0 {
                            continue;
                        }
                        let proj = if self.order > 1 && mls_result.num_neighbors >= 5 * (self.order + 1) * (self.order + 2) / 2 {
                            // 将点沿MLS平面法线投影到多项式表面
                            mls_result.project_point_simple_to_polynomial_surface(u, v)
                        } else {
                            // 将点投影到MLS平面
                            mls_result.project_point_to_mls_plane(u, v)
                        };
                        // 添加投影点及其法线
                        self.add_projected_point_normal(
                            index,
                            &proj.point,
                            &proj.normal,
                            mls_result.curvature,
                            projected_points,
                            projected_points_normals,
                            corresponding_input_indices,
                        );
                        num_added += 1;
                    }
                }
            }
            // 其他上采样方法，目前不做处理
            _ => {}
        }
    }
    
    // 2.22 添加投影点及其法线
    pub fn add_projected_point_normal(
        &self,
        index: usize,
        point: &Vector3<f64>,
        normal: &Vector3<f64>,
        curvature: f32,
        projected_points: &mut Vec<PointXYZRGBNormal>,
        projected_points_normals: &mut Vec<crate::common::Normal>,
        corresponding_input_indices: &mut Vec<usize>,
    ) {
        // TODO: 支持更多类型
        // 创建一个临时的输出点
        let mut aux = PointXYZRGBNormal::default();
        // 将投影点的坐标赋值给临时输出点
        aux.x = point[0] as f32;
        aux.y = point[1] as f32;
        aux.z = point[2] as f32;
    
        // 复制输入点的额外信息到临时输出点
        // self.copy_missing_fields(&self.input[index], &mut aux);
    
        // 将临时输出点添加到投影点集合中
        projected_points.push(aux);
        // 记录该投影点对应的输入点索引
        corresponding_input_indices.push(index);
    
        // 如果需要计算法线
        if self.compute_normals {
            // 创建一个临时的法线对象
            let mut aux_normal = crate::common::Normal::default();
            // 将投影点的法线信息赋值给临时法线对象
            aux_normal.normal[0] = normal[0] as f32;
            aux_normal.normal[1] = normal[1] as f32;
            aux_normal.normal[2] = normal[2] as f32;
            // 将投影点的曲率信息赋值给临时法线对象
            aux_normal.curvature = curvature;
            // 将临时法线对象添加到投影点法线集合中
            projected_points_normals.push(aux_normal);
        }
    }
    
    // 2.23 获取结构体名称
    pub fn get_class_name() -> &'static str {
        "MovingLeastSquares"
    }
    
    // 2.24 Point转Vector3类型
    pub fn point_to_vector3d(point: PointXYZRGBNormal) -> Vector3<f64> {
        Vector3::new(point.x as f64, point.y as f64, point.z as f64)
    }
    
    // 2.25 判断点是否是有限点方法
    fn is_point_finite(point: &PointXYZRGBNormal) -> bool {
        point.x.is_finite() && point.y.is_finite() && point.z.is_finite()
    }
    
}

// 使用 paste 宏生成重复代码，分别为 PointXYZRGB 和 PointXYZRGBNormal 实现 MovingLeastSquares
paste! {
    macro_rules! impl_moving_least_squares {
        ($PointInT:ty, $PointOutT:ty) => {
            impl MovingLeastSquares<$PointInT, $PointOutT> {
                // 复制缺失字段的方法
                pub fn copy_missing_fields(&self, point_in: &$PointInT, point_out: &mut $PointOutT) {
                    // 保存输出点的临时副本
                    let temp = point_out.clone();
                    // TODO 复制输入点的信息到输出点
                    // copy_point(point_in, point_out);
                    // 恢复输出点的 XYZ 坐标
                    point_out.x = temp.x;
                    point_out.y = temp.y;
                    point_out.z = temp.z;
                }
            }
        };
    }

    // 输入输出类型的排列组合
    impl_moving_least_squares!(PointXYZRGB, PointXYZRGB);
    impl_moving_least_squares!(PointXYZRGB, PointXYZRGBNormal);
    impl_moving_least_squares!(PointXYZRGBNormal, PointXYZRGB);
    impl_moving_least_squares!(PointXYZRGBNormal, PointXYZRGBNormal);
}

// 3. PolynomialPartialDerivative的实现
impl PolynomialPartialDerivative{
    // 3.1 构造函数
    fn new() -> Self{
        PolynomialPartialDerivative{
            z: 0.0,
            z_u: 0.0,
            z_v: 0.0,
            z_uu: 0.0,
            z_vv: 0.0,
            z_uv: 0.0,
        }
    }
}


// 4. MLSProjectionResults的实现
impl MLSProjectionResults {
    // 4.1 构造函数
    fn new() -> Self {
        MLSProjectionResults {
            u: 0.0,
            v: 0.0,
            point: Vector3::zeros(),
            normal: Vector3::zeros(),
        }
    }
}

// 5. MLSVoxelGrid的实现
paste! {
    // 使用 paste 宏生成重复代码,
    macro_rules! impl_mlsvoxelgrid {
        ($PointT:ty) => {
            impl MLSVoxelGrid<$PointT> {
                // 5.1 构造函数，用于创建 MLSVoxelGrid 实例
                pub fn new(cloud: &Arc<Vec<$PointT>>, indices: &Vec<usize>, voxel_size: f32, dilation_iteration_num: i32) -> Self {
                    let mut voxel_grid = BTreeMap::new();
                    let mut bounding_min = Vector4::new(f32::MAX, f32::MAX, f32::MAX, 0.0);
                    let mut bounding_max = Vector4::new(f32::MIN, f32::MIN, f32::MIN, 0.0);
                    let mut _marker = PhantomData;

                    // 计算点云的最小和最大边界
                    for &idx in indices.iter() {
                        let point = &cloud[idx];
                        for (i, coord) in [point.x, point.y, point.z].iter().enumerate() {
                            if *coord < bounding_min[i] {
                                bounding_min[i] = *coord;
                            }
                            if *coord > bounding_max[i] {
                                bounding_max[i] = *coord;
                            }
                        }
                    }

                    // 扩展边界以考虑膨胀操作
                    bounding_min -= Vector4::new(voxel_size * (dilation_iteration_num + 1) as f32, voxel_size * (dilation_iteration_num + 1) as f32, voxel_size * (dilation_iteration_num + 1) as f32, 0.0);
                    bounding_max += Vector4::new(voxel_size * (dilation_iteration_num + 1) as f32, voxel_size * (dilation_iteration_num + 1) as f32, voxel_size * (dilation_iteration_num + 1) as f32, 0.0);

                    let bounding_box_size = bounding_max - bounding_min;
                    let max_size = bounding_box_size.x.max(bounding_box_size.y).max(bounding_box_size.z);
                    let data_size = (max_size / voxel_size).ceil() as u64;

                    // 将初始点云放入体素网格
                    for &idx in indices.iter() {
                        let point = &cloud[idx];
                        if point.x.is_finite() {
                            let pos = Self::get_cell_index(&Vector3::new(point.x, point.y, point.z), &bounding_min, voxel_size);
                            let index_1d = Self::get_index_in_1d(&pos, data_size);
                            voxel_grid.insert(index_1d, Leaf::new());
                        }
                    }

                    MLSVoxelGrid {
                        voxel_grid,
                        bounding_min,
                        bounding_max,
                        data_size,
                        voxel_size,
                        _marker,
                    }
                }

                // 5.2 将三维索引转换为一维索引
                pub fn get_index_in_1d(index: &Vector3<i32>, data_size: u64) -> u64 {
                    (index[0] as u64) * data_size * data_size + (index[1] as u64) * data_size + (index[2] as u64)
                }

                // 5.3 将一维索引转换为三维索引
                pub fn get_index_in_3d(index_1d: u64, data_size: u64) -> Vector3<i32> {
                    let mut index_1d_mut = index_1d;
                    let mut index_3d = Vector3::new(0, 0, 0);
                    index_3d[0] = (index_1d_mut / (data_size * data_size)) as i32;
                    index_1d_mut -= (index_3d[0] as u64) * data_size * data_size;
                    index_3d[1] = (index_1d_mut / data_size) as i32;
                    index_1d_mut -= (index_3d[1] as u64) * data_size;
                    index_3d[2] = index_1d_mut as i32;
                    index_3d
                }

                // 5.4 获取点所在的体素索引
                pub fn get_cell_index(p: &Vector3<f32>, bounding_min: &Vector4<f32>, voxel_size: f32) -> Vector3<i32> {
                    let mut index = Vector3::new(0, 0, 0);
                    for i in 0..3 {
                        index[i] = ((p[i] - bounding_min[i]) / voxel_size) as i32;
                    }
                    index
                }

                // 5.5 根据一维索引获取点的位置
                pub fn get_position(index_1d: u64, data_size: u64, bounding_min: &Vector4<f32>, voxel_size: f32) -> Vector3<f32> {
                    let index_3d = Self::get_index_in_3d(index_1d, data_size);
                    let mut point = Vector3::new(0.0, 0.0, 0.0);
                    for i in 0..3 {
                        point[i] = (index_3d[i] as f32) * voxel_size + bounding_min[i];
                    }
                    point
                }

                // 5.6 膨胀操作
                pub fn dilate(&mut self) {
                    let mut new_voxel_grid = self.voxel_grid.clone();
                    for (index_1d, _) in self.voxel_grid.iter() {
                        let index = Self::get_index_in_3d(*index_1d, self.data_size);
                        // 对其所有体素进行膨胀操作
                        for x in -1..=1 {
                            for y in -1..=1 {
                                for z in -1..=1 {
                                    if x != 0 || y != 0 || z != 0 {
                                        let new_index = index + Vector3::new(x, y, z);
                                        let new_index_1d = Self::get_index_in_1d(&new_index, self.data_size);
                                        new_voxel_grid.insert(new_index_1d, Leaf::new());
                                    }
                                }
                            }
                        }
                    }
                    // 更新体素网格
                    self.voxel_grid = new_voxel_grid;
                }
            }
        };
    }

    // 为 PointXYZRGB 实现 MLSVoxelGrid
    impl_mlsvoxelgrid!(PointXYZRGB);
    // 为 PointXYZRGBNormal 实现 MLSVoxelGrid
    impl_mlsvoxelgrid!(PointXYZRGBNormal);
}

// 6. 体素网格中的叶子节点Leaf的实现
impl Leaf {
    // 6.1 默认构造函数
    pub fn new() -> Self {
        Leaf { valid: true }
    }
}

/* end 实现 */

/* start 适配Point */
impl Into<Vector3<f64>> for PointXYZRGBNormal {
    fn into(self) -> Vector3<f64> {
        Vector3::new(self.x as f64, self.y as f64, self.z as f64)
    }
}
/* end 适配Point */

#[cfg(test)]
mod tests1 {
    use super::*;
    use nalgebra::Vector3;

    // 测试 MLSResult 的构造函数
    #[test]
    fn test_mls_result_new() {
        let query_point = Vector3::new(1.0, 2.0, 3.0);
        let mean = Vector3::new(4.0, 5.0, 6.0);
        let plane_normal = Vector3::new(0.0, 0.0, 1.0);
        let u = Vector3::new(1.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 1.0, 0.0);
        let c_vec = vec![1.0, 2.0, 3.0];
        let num_neighbors = 10;
        let curvature = 0.1;
        let order = 2;

        let result = MLSResult::new(
            query_point,
            mean,
            plane_normal,
            u,
            v,
            c_vec.clone(),
            num_neighbors,
            curvature,
            order,
        );

        // 验证构造函数返回的实例的各个字段是否正确
        assert_eq!(result.query_point, query_point);
        assert_eq!(result.mean, mean);
        assert_eq!(result.plane_normal, plane_normal);
        assert_eq!(result.u_axis, u);
        assert_eq!(result.v_axis, v);
        assert_eq!(result.c_vec, c_vec);
        assert_eq!(result.num_neighbors, num_neighbors);
        assert_eq!(result.curvature, curvature);
        assert_eq!(result.order, order);
        assert!(result.valid);
    }

    // 测试 get_mls_coordinates 方法
    #[test]
    fn test_get_mls_coordinates() {
        let query_point = Vector3::new(0.0, 0.0, 0.0);
        let mean = Vector3::new(0.0, 0.0, 0.0);
        let plane_normal = Vector3::new(0.0, 0.0, 1.0);
        let u = Vector3::new(1.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 1.0, 0.0);
        let c_vec = vec![1.0];
        let num_neighbors = 1;
        let curvature = 0.0;
        let order = 1;

        let result = MLSResult::new(
            query_point,
            mean,
            plane_normal,
            u,
            v,
            c_vec,
            num_neighbors,
            curvature,
            order,
        );

        let pt = Vector3::new(1.0, 2.0, 3.0);
        let (u_coord, v_coord, w_coord) = result.get_mls_coordinates(&pt);

        // 验证计算得到的 u, v, w 坐标是否正确
        assert_eq!(u_coord, pt.dot(&u));
        assert_eq!(v_coord, pt.dot(&v));
        assert_eq!(w_coord, pt.dot(&plane_normal));
    }

    // 测试 get_polynomial_value 方法
    #[test]
    fn test_get_polynomial_value() {
        let query_point = Vector3::new(0.0, 0.0, 0.0);
        let mean = Vector3::new(0.0, 0.0, 0.0);
        let plane_normal = Vector3::new(0.0, 0.0, 1.0);
        let u = Vector3::new(1.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 1.0, 0.0);
        let c_vec = vec![1.0];
        let num_neighbors = 1;
        let curvature = 0.0;
        let order = 1;

        let result = MLSResult::new(
            query_point,
            mean,
            plane_normal,
            u,
            v,
            c_vec,
            num_neighbors,
            curvature,
            order,
        );

        let u = 1.0;
        let v = 2.0;
        let value = result.get_polynomial_value(u, v);

        // 验证计算得到的多项式值是否正确
        assert_eq!(value, 1.0);
    }

    // 测试 get_polynomial_partial_derivative 方法
    #[test]
    fn test_get_polynomial_partial_derivative() {
        let query_point = Vector3::new(0.0, 0.0, 0.0);
        let mean = Vector3::new(0.0, 0.0, 0.0);
        let plane_normal = Vector3::new(0.0, 0.0, 1.0);
        let u = Vector3::new(1.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 1.0, 0.0);
        let c_vec = vec![1.0];
        let num_neighbors = 1;
        let curvature = 0.0;
        let order = 1;

        let result = MLSResult::new(
            query_point,
            mean,
            plane_normal,
            u,
            v,
            c_vec,
            num_neighbors,
            curvature,
            order,
        );

        let u = 1.0;
        let v = 2.0;
        let d = result.get_polynomial_partial_derivative(u, v);

        // 验证计算得到的多项式偏导数是否正确
        assert_eq!(d.z, 1.0);
        assert_eq!(d.z_u, 0.0);
        assert_eq!(d.z_v, 0.0);
        assert_eq!(d.z_uu, 0.0);
        assert_eq!(d.z_vv, 0.0);
        assert_eq!(d.z_uv, 0.0);
    }

    // 测试 calculate_principal_curvatures 方法
    #[test]
    fn test_calculate_principal_curvatures() {
        let query_point = Vector3::new(0.0, 0.0, 0.0);
        let mean = Vector3::new(0.0, 0.0, 0.0);
        let plane_normal = Vector3::new(0.0, 0.0, 1.0);
        let u = Vector3::new(1.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 1.0, 0.0);
        let c_vec = vec![1.0];
        let num_neighbors = 1;
        let curvature = 0.0;
        let order = 1;

        let result = MLSResult::new(
            query_point,
            mean,
            plane_normal,
            u,
            v,
            c_vec,
            num_neighbors,
            curvature,
            order,
        );

        let u = 1.0;
        let v = 2.0;
        let k = result.calculate_principal_curvatures(u, v);

        // 验证计算得到的主曲率是否正确
        assert_eq!(k[0], 1e-5 as f32);
        assert_eq!(k[1], 1e-5 as f32);
    }

    // 测试 compute_mls_weight 方法
    #[test]
    fn test_compute_mls_weight() {
        let query_point = Vector3::new(0.0, 0.0, 0.0);
        let mean = Vector3::new(0.0, 0.0, 0.0);
        let plane_normal = Vector3::new(0.0, 0.0, 1.0);
        let u = Vector3::new(1.0, 0.0, 0.0);
        let v = Vector3::new(0.0, 1.0, 0.0);
        let c_vec = vec![1.0];
        let num_neighbors = 1;
        let curvature = 0.0;
        let order = 1;

        let result = MLSResult::new(
            query_point,
            mean,
            plane_normal,
            u,
            v,
            c_vec,
            num_neighbors,
            curvature,
            order,
        );

        let sq_dist = 1.0;
        let sq_mls_radius = 2.0;
        let weight = result.compute_mls_weight(sq_dist, sq_mls_radius);

        // 验证计算得到的 MLS 权重是否正确
        assert_eq!(weight, std::f64::consts::E.powf(-sq_dist / sq_mls_radius));
    }
}

#[cfg(test)]
mod tests2 {
    use super::*;
    use nalgebra::Vector3;
    use rand::{SeedableRng,rng};

    // 测试 MovingLeastSquares 的构造函数
    #[test]
    fn test_moving_least_squares_new() {
        let mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();

        // 验证构造函数初始化的字段是否符合默认值
        assert!(mls.input.is_empty());
        assert!(mls.output.is_empty());
        assert_eq!(mls.search_radius, 0.0);
        assert_eq!(mls.order, 2);
        assert!(!mls.compute_normals);
        assert!(mls.mls_results.is_empty());
        assert!(mls.distinct_cloud.is_empty());
        assert_eq!(mls.upsample_method, UpsamplingMethod::NONE);
        assert_eq!(mls.upsampling_radius, 0.0);
        assert_eq!(mls.upsampling_step, 0.0);
        assert_eq!(mls.desired_num_points_in_radius, 0);
        assert!(mls.cache_mls_results);
        assert_eq!(mls.projection_method, ProjectionMethod::SIMPLE);
        assert_eq!(mls.threads, 1);
        assert_eq!(mls.voxel_size, 1.0);
        assert_eq!(mls.dilation_iteration_num, 0);
        assert!(mls.corresponding_input_indices.is_empty());
        assert!(mls.rng_uniform_distribution.is_none());
    }

    // 测试 set_input_cloud 方法
    #[test]
    fn test_set_input_cloud() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let cloud = Arc::new(vec![PointXYZRGBNormal::default()]);
        mls.set_input_cloud(cloud.clone());

        // 验证设置输入点云后，input 字段是否正确更新
        assert_eq!(mls.input, cloud);
    }

    // 测试 set_search_radius 方法
    #[test]
    fn test_set_search_radius() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let radius = 1.0;
        mls.set_search_radius(radius);

        // 验证设置搜索半径后，search_radius 字段是否正确更新
        assert_eq!(mls.search_radius, radius);
    }

    // 测试 set_polynomial_order 方法
    #[test]
    fn test_set_polynomial_order() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let order = 3;
        mls.set_polynomial_order(order);

        // 验证设置多项式阶数后，order 字段是否正确更新
        assert_eq!(mls.order, order);
    }

    // 测试 set_compute_normals 方法
    #[test]
    fn test_set_compute_normals() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let compute = true;
        mls.set_compute_normals(compute);

        // 验证设置是否计算法线后，compute_normals 字段是否正确更新
        assert_eq!(mls.compute_normals, compute);
    }

    // 测试 set_distinct_cloud 方法
    #[test]
    fn test_set_distinct_cloud() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let cloud = Arc::new(vec![PointXYZRGBNormal::default()]);
        mls.set_distinct_cloud(cloud.clone());

        // 验证设置不同点云后，distinct_cloud 字段是否正确更新
        assert_eq!(mls.distinct_cloud, cloud);
    }

    // 测试 set_upsampling_method 方法
    #[test]
    fn test_set_upsampling_method() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let method = UpsamplingMethod::DISTINCT_CLOUD;
        mls.set_upsampling_method(method);

        // 验证设置上采样方法后，upsample_method 字段是否正确更新
        assert_eq!(mls.upsample_method, method);
    }

    // 测试 set_upsampling_radius 方法
    #[test]
    fn test_set_upsampling_radius() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let radius = 1.0;
        mls.set_upsampling_radius(radius);

        // 验证设置上采样半径后，upsampling_radius 字段是否正确更新
        assert_eq!(mls.upsampling_radius, radius);
    }

    // 测试 set_upsampling_step_size 方法
    #[test]
    fn test_set_upsampling_step_size() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let step_size = 0.1;
        mls.set_upsampling_step_size(step_size);

        // 验证设置上采样步长后，upsampling_step 字段是否正确更新
        assert_eq!(mls.upsampling_step, step_size);
    }

    // 测试 set_point_density 方法
    #[test]
    fn test_set_point_density() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let num_points = 10;
        mls.set_point_density(num_points);

        // 验证设置搜索半径内期望点数后，desired_num_points_in_radius 字段是否正确更新
        assert_eq!(mls.desired_num_points_in_radius, num_points);
    }

    // 测试 set_cache_mls_results 方法
    #[test]
    fn test_set_cache_mls_results() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let cache = false;
        mls.set_cache_mls_results(cache);

        // 验证设置是否缓存 MLS 结果后，cache_mls_results 字段是否正确更新
        assert_eq!(mls.cache_mls_results, cache);
    }

    // 测试 set_projection_method 方法
    #[test]
    fn test_set_projection_method() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let method = ProjectionMethod::ORTHOGONAL;
        mls.set_projection_method(method);

        // 验证设置投影方法后，projection_method 字段是否正确更新
        assert_eq!(mls.projection_method, method);
    }

    // 测试 set_number_of_threads 方法
    #[test]
    fn test_set_number_of_threads() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let threads = 4;
        mls.set_number_of_threads(threads);

        // 验证设置最大线程数后，threads 字段是否正确更新
        assert_eq!(mls.threads, threads);
    }

    // 测试 set_dilation_voxel_size 方法
    #[test]
    fn test_set_dilation_voxel_size() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let size = 0.5;
        mls.set_dilation_voxel_size(size);

        // 验证设置体素大小后，voxel_size 字段是否正确更新
        assert_eq!(mls.voxel_size, size);
    }

    // 测试 set_dilation_iterations 方法
    #[test]
    fn test_set_dilation_iterations() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let iterations = 2;
        mls.set_dilation_iterations(iterations);

        // 验证设置体素网格膨胀迭代次数后，dilation_iteration_num 字段是否正确更新
        assert_eq!(mls.dilation_iteration_num, iterations);
    }

    // 测试 find_neighbors 方法
    #[test]
    fn test_find_neighbors() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let cloud = vec![
            PointXYZRGBNormal { x: 0.0, y: 0.0, z: 0.0, ..Default::default() },
            PointXYZRGBNormal { x: 1.0, y: 0.0, z: 0.0, ..Default::default() },
            PointXYZRGBNormal { x: 2.0, y: 0.0, z: 0.0, ..Default::default() },
        ];
        mls.input = Arc::new(cloud);
        let index = 0;
        let radius = 1.5;
        let neighbors = mls.find_neighbors(index, radius);

        // 验证查找邻居方法返回的邻居索引是否正确
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
        assert!(!neighbors.contains(&2));
    }

    // 测试 find_neighbors_cloud 方法
    #[test]
    fn test_find_neighbors_cloud() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let cloud = vec![
            PointXYZRGBNormal { x: 0.0, y: 0.0, z: 0.0, ..Default::default() },
            PointXYZRGBNormal { x: 1.0, y: 0.0, z: 0.0, ..Default::default() },
            PointXYZRGBNormal { x: 2.0, y: 0.0, z: 0.0, ..Default::default() },
        ];
        mls.input = Arc::new(cloud);
        let point = PointXYZRGBNormal { x: 0.5, y: 0.0, z: 0.0, ..Default::default() };
        let k = 1;
        let neighbors = mls.find_neighbors_cloud(&point, k);

        // 验证在不同点云中查找邻居方法返回的邻居索引是否正确
        assert_eq!(neighbors.len(), k);
        assert!(neighbors.contains(&0));
    }

    // 测试 process 方法
    #[test]
    fn test_process() {
        let mut mls = MovingLeastSquares::<PointXYZRGBNormal, PointXYZRGBNormal>::new();
        let cloud = vec![
            PointXYZRGBNormal { x: 0.0, y: 0.0, z: 0.0, ..Default::default() },
        ];
        mls.input = Arc::new(cloud);
        mls.search_radius = 1.0;
        mls.rng = rng();

        mls.process();

        // 验证处理方法执行后，输出点云是否有更新
        assert!(!mls.output.is_empty());
    }
}

#[cfg(test)]
mod tests5 {
    use super::*;
    use nalgebra::{Vector3, Vector4};

    // 测试 MLSVoxelGrid 的构造函数
    #[test]
    fn test_mlsvoxelgrid_new() {
        let cloud = Arc::new(vec![
            PointXYZRGBNormal { x: 0.0, y: 0.0, z: 0.0, ..Default::default() },
            PointXYZRGBNormal { x: 1.0, y: 1.0, z: 1.0, ..Default::default() },
        ]);
        let indices = (0..cloud.len()).collect();
        let voxel_size = 0.5;
        let dilation_iteration_num = 1;

        let voxel_grid = MLSVoxelGrid::<PointXYZRGBNormal>::new(&cloud, &indices, voxel_size, dilation_iteration_num);

        // 验证体素网格的基本属性是否正确设置
        assert!(!voxel_grid.voxel_grid.is_empty());
        assert_ne!(voxel_grid.bounding_min, Vector4::new(f32::MAX, f32::MAX, f32::MAX, 0.0));
        assert_ne!(voxel_grid.bounding_max, Vector4::new(f32::MIN, f32::MIN, f32::MIN, 0.0));
        assert!(voxel_grid.data_size > 0);
        assert_eq!(voxel_grid.voxel_size, voxel_size);
    }

    // 测试 get_index_in_1d 方法
    #[test]
    fn test_get_index_in_1d() {
        let index = Vector3::new(1, 2, 3);
        let data_size = 10;
        let index_1d = MLSVoxelGrid::<PointXYZRGBNormal>::get_index_in_1d(&index, data_size);

        // 验证三维索引转换为一维索引的计算结果是否正确
        assert_eq!(index_1d, (1 as u64) * data_size * data_size + (2 as u64) * data_size + (3 as u64));
    }

    // 测试 get_index_in_3d 方法
    #[test]
    fn test_get_index_in_3d() {
        let index_1d = 123;
        let data_size = 10;
        let index_3d = MLSVoxelGrid::<PointXYZRGBNormal>::get_index_in_3d(index_1d, data_size);

        let mut expected_index_3d = Vector3::new(0, 0, 0);
        expected_index_3d[0] = (index_1d / (data_size * data_size)) as i32;
        let remaining = index_1d - (expected_index_3d[0] as u64) * data_size * data_size;
        expected_index_3d[1] = (remaining / data_size) as i32;
        expected_index_3d[2] = (remaining % data_size) as i32;

        // 验证一维索引转换为三维索引的计算结果是否正确
        assert_eq!(index_3d, expected_index_3d);
    }

    // 测试 get_cell_index 方法
    #[test]
    fn test_get_cell_index() {
        let p = Vector3::new(1.2, 2.3, 3.4);
        let bounding_min = Vector4::new(0.0, 0.0, 0.0, 0.0);
        let voxel_size = 1.0;
        let index = MLSVoxelGrid::<PointXYZRGBNormal>::get_cell_index(&p, &bounding_min, voxel_size);

        // 验证获取点所在体素索引的计算结果是否正确
        assert_eq!(index, Vector3::new(1, 2, 3));
    }

    // 测试 get_position 方法
    #[test]
    fn test_get_position() {
        let index_1d = 123;
        let data_size = 10;
        let bounding_min = Vector4::new(0.0, 0.0, 0.0, 0.0);
        let voxel_size = 1.0;
        let position = MLSVoxelGrid::<PointXYZRGBNormal>::get_position(index_1d, data_size, &bounding_min, voxel_size);

        let index_3d = MLSVoxelGrid::<PointXYZRGBNormal>::get_index_in_3d(index_1d, data_size);
        let mut expected_position = Vector3::new(0.0, 0.0, 0.0);
        for i in 0..3 {
            expected_position[i] = (index_3d[i] as f32) * voxel_size + bounding_min[i];
        }

        // 验证根据一维索引获取点位置的计算结果是否正确
        assert_eq!(position, expected_position);
    }

    // 测试 dilate 方法
    #[test]
    fn test_dilate() {
        let cloud = Arc::new(vec![
            PointXYZRGBNormal { x: 0.0, y: 0.0, z: 0.0, ..Default::default() },
        ]);
        let indices = (0..cloud.len()).collect();
        let voxel_size = 0.5;
        let dilation_iteration_num = 1;

        let mut voxel_grid = MLSVoxelGrid::<PointXYZRGBNormal>::new(&cloud, &indices, voxel_size, dilation_iteration_num);
        let original_size = voxel_grid.voxel_grid.len();

        voxel_grid.dilate();

        // 验证膨胀操作后体素网格的大小是否增加
        assert!(voxel_grid.voxel_grid.len() > original_size);
    }
}
