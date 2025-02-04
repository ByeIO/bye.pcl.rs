#![allow(unused_macros)]
#![allow(unused_unsafe)]
#![allow(non_camel_case_types)]

//! 点云结构体描述符大小

/* start 点云描述符 */
// 定义所有具有descriptorSize()成员函数的点类型
const PCL_DESCRIPTOR_FEATURE_POINT_TYPES: &[&str] = &[
    "PFHSignature125",
    "PFHRGBSignature250",
    "FPFHSignature33",
    "VFHSignature308",
    "GASDSignature512",
    "GASDSignature984",
    "GASDSignature7992",
    "GRSDSignature21",
    "ESFSignature640",
    "BRISKSignature512",
    "Narf36",
];

// 定义描述符大小的结构体
struct DescriptorSize;

impl DescriptorSize {
    fn value<T: AsRef<str>>(type_name: T) -> i32 {
        match type_name.as_ref() {
            "PFHSignature125" => 125,
            "PFHRGBSignature250" => 250,
            "ShapeContext1980" => 1980,
            "UniqueShapeContext1960" => 1960,
            "SHOT352" => 352,
            "SHOT1344" => 1344,
            "FPFHSignature33" => 33,
            "VFHSignature308" => 308,
            "GRSDSignature21" => 21,
            "BRISKSignature512" => 512,
            "ESFSignature640" => 640,
            "GASDSignature512" => 512,
            "GASDSignature984" => 984,
            "GASDSignature7992" => 7992,
            "GFPFHSignature16" => 16,
            "Narf36" => 36,
            _ => panic!("Unknown feature point type"),
        }
    }
}

// 定义描述符大小的常量
pub fn descriptor_size_v<T: AsRef<str>>(type_name: T) -> i32 {
    DescriptorSize::value(type_name)
}
/* end 点云描述符 */

#[cfg(test)]
mod tests1 {
    use super::*;

    // 测试描述符大小的计算
    #[test]
    fn test_descriptor_size() {
        // 测试 PFHSignature125 的描述符大小
        assert_eq!(descriptor_size_v("PFHSignature125"), 125);

        // 测试 PFHRGBSignature250 的描述符大小
        assert_eq!(descriptor_size_v("PFHRGBSignature250"), 250);

        // 测试 FPFHSignature33 的描述符大小
        assert_eq!(descriptor_size_v("FPFHSignature33"), 33);

        // 测试 VFHSignature308 的描述符大小
        assert_eq!(descriptor_size_v("VFHSignature308"), 308);

        // 测试 GRSDSignature21 的描述符大小
        assert_eq!(descriptor_size_v("GRSDSignature21"), 21);

        // 测试 BRISKSignature512 的描述符大小
        assert_eq!(descriptor_size_v("BRISKSignature512"), 512);

        // 测试 ESFSignature640 的描述符大小
        assert_eq!(descriptor_size_v("ESFSignature640"), 640);

        // 测试 GASDSignature512 的描述符大小
        assert_eq!(descriptor_size_v("GASDSignature512"), 512);

        // 测试 GASDSignature984 的描述符大小
        assert_eq!(descriptor_size_v("GASDSignature984"), 984);

        // 测试 GASDSignature7992 的描述符大小
        assert_eq!(descriptor_size_v("GASDSignature7992"), 7992);

        // 测试 Narf36 的描述符大小
        assert_eq!(descriptor_size_v("Narf36"), 36);
    }

    // 测试未知类型的描述符大小
    #[test]
    #[should_panic(expected = "Unknown feature point type")]
    fn test_unknown_descriptor_size() {
        descriptor_size_v("UnknownType");
    }
}
