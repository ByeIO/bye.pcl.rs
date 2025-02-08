//! 将point_types.h, point_types.hpp, point_types.cpp拆分为多个文件

// 1. 点结构体定义, 向上一级代码导出
pub mod defines;
pub use defines::{
    Axis,
    BRISKSignature512,
    BorderDescription,
    BorderTraits,
    Boundary,
    CPPFSignature,
    ESFSignature640,
    FPFHSignature33,
    GASDSignature512,
    GASDSignature7992,
    GASDSignature984,
    GFPFHSignature16,
    GRSDSignature21,
    Histogram,
    Intensity,
    Intensity32u,
    Intensity8u,
    IntensityGradient,
    InterestPoint,
    Label,
    MomentInvariants,
    Narf36,
    Normal,
    NormalBasedSignature12,
    PFHRGBSignature250,
    PFHSignature125,
    PPFRGBSignature,
    PPFSignature,
    PointDEM,
    PointNormal,
    PointSurfel,
    PointUV,
    PointWithRange,
    PointWithScale,
    PointWithViewpoint,
    PointXY,
    PointXYZ,
    PointXYZHSV,
    PointXYZI,
    PointXYZINormal,
    PointXYZL,
    PointXYZLAB,
    PointXYZLNormal,
    PointXYZRGB,
    PointXYZRGBA,
    PointXYZRGBL,
    PointXYZRGBNormal,
    PrincipalCurvatures,
    PrincipalRadiiRSD,
    ReferenceFrame,
    ShapeContext1980,
    UniqueShapeContext1960,
    VFHSignature308,
    RGB,
    SHOT1344,
    SHOT352,
    // 枚举
    PCL_DESCRIPTOR_FEATURE_POINT_TYPES,
    PCL_FEATURE_POINT_TYPES,
    PCL_NORMAL_POINT_TYPES,
    PCL_XYZL_POINT_TYPES,
};

// 2. 点结构体描述符大小, 向上一级代码导出
pub mod descriptor_size;
pub use descriptor_size::descriptor_size_v;

// 3. 适配到nalgebra矩阵, 向上一级代码导出
pub mod eigen_map;
pub use eigen_map::{
    Array3fMap, Array3fMapConst, Array4fMap, Array4fMapConst, PclAddIntensity, PclAddIntensity32u,
    PclAddIntensity8u, PclAddNormal4D, PclAddPoint4D, PclAddRGB, Vector2fMap, Vector2fMapConst,
    Vector3c, Vector3cMap, Vector3cMapConst, Vector3fMap, Vector3fMapConst, Vector4c, Vector4cMap,
    Vector4cMapConst, Vector4fMap, Vector4fMapConst,
};

// 4. 实现点构造函数(移除了与内存对齐相关的逻辑)
pub mod impls;
// pub use impls::*;

// 5. 实现点属性存在与否的判断
pub mod has_sth;
pub use has_sth::*;

#[cfg(test)]
mod tests1 {
    use super::*;

    //  测试PointXYZ的has_xyz()
    #[test]
    fn test_point_xyz() {
        let p1 = PointXYZ::new();
        assert_eq!(p1.x, 0.0);
        assert_eq!(p1.y, 0.0);
        assert_eq!(p1.z, 0.0);

        // assert!(p1.has_xyz());
    }
}
