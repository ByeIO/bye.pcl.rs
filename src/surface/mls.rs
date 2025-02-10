#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]
#![allow(unused_imports)]

//! 最小二乘搜索法

// 标准库
use std::collections::HashMap;
use std::sync::Arc;

// 线性代数
use nalgebra::{
    Vector2, Vector3, Vector4, 
    Matrix2, Matrix3, Matrix, Vector, Const,
    DVector, DMatrix, DefaultAllocator,
    allocator::Allocator, 
};

/* start 结构体 */
// 1. 定义 MLSResult 结构体，用于存储 MLS 拟合的结果
#[derive(Default)]
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

/* end 结构体 */

/* start 枚举 */

// 1. 投影方法
#[derive(Clone, Copy, PartialEq)]
pub enum ProjectionMethod{
    // 投影到MLS平面。
    NONE,      
    // 沿MLS平面的法线投影到多项式表面。
    SIMPLE,    
    // 投影到多项式表面上最近的点。
    ORTHOGONAL,
}

// 2. 上采样方法
#[derive(Clone, Copy, PartialEq)]
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

    // // 处理点云
    // pub fn process(&mut self) {
    //     // 处理逻辑...
    //     for index in 0..self.input.len() {
    //         // 计算邻居
    //         let nn_indices = self.find_neighbors(index);
    //         // 计算 MLS 表面
    //         self.compute_mls_surface(index, &nn_indices);
    //     }
    // }

    // // 查找邻居
    // fn find_neighbors(&self, index: usize) -> Vec<usize> {
    //     // 邻居查找逻辑...
    //     vec![] // 返回邻居索引
    // }

    // // 计算 MLS 表面
    // fn compute_mls_surface(&mut self, index: usize, nn_indices: &[usize]) {
    //     // 计算 MLS 逻辑...
    //     // 查询点
    //     let query_point = self.input[index]; 
    //     let mls_result_default = MLSResult::default();
    //     let _ = mls_result_default.query_point = query_point;
    //     let mls_result = mls_result_default;
    //     self.mls_results.push(mls_result);
    // }

    // // 计算法线
    // fn compute_normals(&self) {
    //     if self.compute_normals {
    //         // 法线计算逻辑...
    //     }
    // }

    // // 投影点到 MLS 表面
    // pub fn project_point(&self, point: &Vector3<f64>) -> Vector3<f64> {
    //     // 投影逻辑...
    //     *point // 返回投影后的点
    // }

    // 其他方法...
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

/* end 实现 */
