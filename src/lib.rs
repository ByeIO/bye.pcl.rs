#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
#![allow(non_fmt_panics)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![allow(rustdoc::missing_crate_level_docs)]
#![allow(unsafe_code)]
#![allow(clippy::undocumented_unsafe_blocks)]
#![allow(unused_must_use)]
#![allow(non_snake_case)]
#![allow(unused_doc_comments)]
#![feature(const_trait_impl)]
#![feature(trivial_bounds)]
// #![feature(effects)]
// unsafe
// #![allow(incomplete_features)]

// 1. 2d模块
pub mod _2d;

// 2. common模块
pub mod common;
pub use f3l_core;

// 3. features模块
pub mod features;
pub use f3l_features;

//4. filters模块
pub mod filters;
pub use f3l_filter;

// 5. geometry模块
pub mod geometry;

// 6. io模块
pub mod io;

// 7. kdtree模块: KD树数据结构
pub mod kdtree;
pub use f3l_search_tree;

// 8. keypoints模块: 关键点检测
pub mod keypoints;

// 9. ml模块: 机器学习
pub mod ml;

// 10. octree模块: 八叉树数据结构
pub mod octree;
// pub use f3l_search_tree;

// 11. outofcore模块
pub mod outofcore;

// 12. people模块
pub mod people;

// 13. recognition模块
pub mod recognition;

// 14. registration模块
pub mod registration;

// 15. sample_consensus模块
pub mod sample_consensus;

// 16. search模块
pub mod search;

// 17. segmentation模块
pub mod segmentation;
pub use f3l_segmentation;

// 18. simulation模块
pub mod simulation;

// 19. stereo模块
pub mod stereo;

// 20. surface模块
pub mod surface;
pub use f3l_surface;

// 21. tools模块
pub mod tools;

// 22. tracking模块
pub mod tracking;

// 23. visualization模块
pub mod visualization;
