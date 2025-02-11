use std::fmt;
use std::sync::Arc;

/// PCLHeader 结构体，用于存储点云数据的头部信息
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PclHeader {
    /// 序列号
    pub seq: u32,
    /// 时间戳，表示从1970-01-01 00:00:00（UNIX纪元）开始的微秒数
    pub stamp: u64,
    /// 坐标系ID
    pub frame_id: String,
}

/// PCLHeader 的智能指针类型
pub type HeaderPtr = Arc<PclHeader>;
pub type HeaderConstPtr = Arc<PclHeader>;

/// 实现构造函数
impl PclHeader {
    // 手动创建, 需要输入所有参数
    pub fn new(_seq: u32, _stamp: u64, _frame_id: String) -> Self {
        PclHeader {
            seq: _seq,
            stamp: _stamp,
            frame_id: _frame_id,
        }
    }
    // 自动添加时间和序列号的构造函数
    pub fn new_auto() -> Self {
        PclHeader {
            seq: rand::random(),
            stamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            frame_id: String::from(""),
        }
    }
}

/// 实现 Display trait 用于格式化输出
impl fmt::Display for PclHeader {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "seq: {} stamp: {} frame_id: {}",
            self.seq, self.stamp, self.frame_id
        )
    }
}

// /// 实现 PartialEq trait 用于比较两个 PCLHeader 是否相等
// impl PartialEq for PclHeader {
//     fn eq(&self, other: &Self) -> bool {
//         self.seq == other.seq && self.stamp == other.stamp && self.frame_id == other.frame_id
//     }
// }
