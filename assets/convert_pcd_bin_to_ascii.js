import * as PCL from "./pcljs/pcl.js"

// 自动加载pcl.js同目录的pcl-core.wasm
await PCL.init();

var cloud = PCL.loadPCDFile("./office1.pcd", PCL.PoiintXYZRGB);
savePCDFileASCII("./office1_ascii.pcd", cloud);
