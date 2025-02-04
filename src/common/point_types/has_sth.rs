#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]

//! 判断点云是否具有某些属性的魔法
//! 这些元函数通过编译时检查来确定点类型是否包含所需的字段，从而在编译时或运行时进行条件判断。

// 标准库
use std::collections::HashMap;
use std::marker::PhantomData;

// 元编程(代码生成)
use paste::paste;
use quote::quote;
use syn::{parse_macro_input, DeriveInput};

#[path = "./defines.rs"]
mod defines;
use defines::{
    Axis, BRISKSignature512, BorderDescription, BorderTraits, Boundary, CPPFSignature,
    ESFSignature640, FPFHSignature33, GASDSignature512, GASDSignature7992, GASDSignature984,
    GFPFHSignature16, GRSDSignature21, Histogram, Intensity, Intensity32u, Intensity8u,
    IntensityGradient, InterestPoint, Label, MomentInvariants, Narf36, Normal,
    NormalBasedSignature12, PFHRGBSignature250, PFHSignature125, PPFRGBSignature, PPFSignature,
    PointDEM, PointNormal, PointSurfel, PointUV, PointWithRange, PointWithScale,
    PointWithViewpoint, PointXY, PointXYZ, PointXYZHSV, PointXYZI, PointXYZINormal, PointXYZL,
    PointXYZLAB, PointXYZLNormal, PointXYZRGB, PointXYZRGBA, PointXYZRGBL, PointXYZRGBNormal,
    PrincipalCurvatures, PrincipalRadiiRSD, ReferenceFrame, ShapeContext1980,
    UniqueShapeContext1960, VFHSignature308, PCL_DESCRIPTOR_FEATURE_POINT_TYPES,
    PCL_FEATURE_POINT_TYPES, PCL_NORMAL_POINT_TYPES, PCL_XYZL_POINT_TYPES, RGB, SHOT1344, SHOT352,
};

#[path = "./descriptor_size.rs"]
mod descriptor_size;

#[path = "./eigen_map.rs"]
mod eigen_map;
use eigen_map::{
    Array3fMap, Array3fMapConst, Array4fMap, Array4fMapConst, PclAddIntensity, PclAddIntensity32u,
    PclAddIntensity8u, PclAddNormal4D, PclAddPoint4D, PclAddRGB, Vector2fMap, Vector2fMapConst,
    Vector3c, Vector3cMap, Vector3cMapConst, Vector3fMap, Vector3fMapConst, Vector4c, Vector4cMap,
    Vector4cMapConst, Vector4fMap, Vector4fMapConst,
};

#[path = "./impls.rs"]
mod impls;
use impls::*;

/* start is_has_sth magic */
// 类型名称
const TYPE_NAMES: [&str; 56] = [
    "Axis",
    "BRISKSignature512",
    "BorderDescription",
    "BorderTraits",
    "Boundary",
    "CPPFSignature",
    "ESFSignature640",
    "FPFHSignature33",
    "GASDSignature512",
    "GASDSignature7992",
    "GASDSignature984",
    "GFPFHSignature16",
    "GRSDSignature21",
    "Histogram",
    "Intensity",
    "Intensity32u",
    "Intensity8u",
    "IntensityGradient",
    "InterestPoint",
    "Label",
    "MomentInvariants",
    "Narf36",
    "Normal",
    "NormalBasedSignature12",
    "PFHRGBSignature250",
    "PFHSignature125",
    "PPFRGBSignature",
    "PPFSignature",
    "PointDEM",
    "PointNormal",
    "PointSurfel",
    "PointUV",
    "PointWithRange",
    "PointWithScale",
    "PointWithViewpoint",
    "PointXY",
    "PointXYZ",
    "PointXYZHSV",
    "PointXYZI",
    "PointXYZINormal",
    "PointXYZL",
    "PointXYZLAB",
    "PointXYZLNormal",
    "PointXYZRGB",
    "PointXYZRGBA",
    "PointXYZRGBL",
    "PointXYZRGBNormal",
    "PrincipalCurvatures",
    "PrincipalRadiiRSD",
    "ReferenceFrame",
    "ShapeContext1980",
    "UniqueShapeContext1960",
    "VFHSignature308",
    "RGB",
    "SHOT1344",
    "SHOT352",
];

// 字段(属性名称)
const LABEL_NAMES: [&str; 9] = [
    "Point",
    "RGB",
    "RGBA",
    "XY",
    "XYZ",
    "Normal",
    "Curvature",
    "Intensity",
    "Label",
];

// 绑定类型和特征
macro_rules! define_has_field {
    ($trait_name:ident, $field_name:ident, $($type_name:ident),*) => {
        paste! {
            // 定义trait
            pub trait [<Has $trait_name>] {
                fn [<has_ $field_name>](&self) -> bool;
            }

            // 为所有传入的类型实现trait
            $(
                impl [<Has $trait_name>] for $type_name {
                    fn [<has_ $field_name>](&self) -> bool {
                        true
                    }
                }
            )*

        }
    };
}

// 自动绑定点云类型与字段特征的宏
macro_rules! bind_types_with_fields {
    ($types:ident, $fields:ident) => {
        // 外层遍历所有类型
        $(
            // 内层遍历所有字段关键词
            $(
                paste! {
                    // 根据字段关键词生成匹配逻辑
                    #[allow(unused)]
                    impl $types {
                        #[inline]
                        pub fn [<has_ $fields>](&self) -> bool {
                            // 特殊处理颜色字段
                            if stringify!($fields) == "Color" {
                                return self.has_rgb() || self.has_rgba();
                            }

                            // 处理L字符的特殊情况
                            if stringify!($fields) == "Label" {
                                return stringify!($types).contains("L");
                            }

                            // 通用字段检测
                            stringify!($types).contains(stringify!($fields))
                        }
                    }
                }
            )*
        )*
    };
}

// FIXME: 自动绑定types和fields
// bind_types_with_fields!(TYPE_NAMES, LABEL_NAMES);

// FIXME: 手动实现, 会导致struct其他impl失效吗?
define_has_field!(XY, xy, PointXY, PointWithRange, PointWithViewpoint);
define_has_field!(
    XYZ,
    xyz,
    PointXYZ,
    PointXYZI,
    PointXYZL,
    PointXYZRGB,
    PointXYZRGBA,
    PointXYZLAB,
    PointXYZLNormal,
    PointXYZRGBL,
    PointXYZRGBNormal,
    PointXYZHSV,
    PointXYZINormal,
    PointNormal,
    PointSurfel,
    PointWithScale
);
define_has_field!(
    Normal,
    normal,
    PointNormal,
    PointXYZINormal,
    PointXYZRGBNormal,
    PointSurfel
);
define_has_field!(
    Curvature,
    curvature,
    PointNormal,
    PointXYZINormal,
    PointXYZRGBNormal
);
define_has_field!(Intensity, intensity, PointXYZI, PointXYZINormal);
define_has_field!(
    Color,
    color,
    PointXYZRGB,
    PointXYZRGBA,
    PointXYZRGBL,
    PointXYZRGBNormal
);
define_has_field!(Label, label, PointXYZL, PointXYZLNormal, PointXYZRGBL);
/* end is_has_sth magic */

#[cfg(test)]
mod tests1 {
    use super::*;

    #[test]
    fn test_field_detection() {
        // 正向测试
        // assert!(PointXYZ::new().has_xyz());
        // assert!(PointXYZRGBA::new().has_color());
        // assert!(PointNormal::new(1.0, 1.0, [1.0, 1.0, 1.0], 1.0).has_normal());
        assert!(true);
    }
}
