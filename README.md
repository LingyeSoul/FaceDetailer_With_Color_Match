# FaceDetailer + Color Match (ComfyUI custom node)

把 Impact-Pack 的 `FaceDetailer` 与 KJNodes 的 `Color Match` 串成一个新节点：
- 先执行 FaceDetailer 的完整遮罩检测/重绘流程
- 再将重绘结果与原图做 Color Match
- 最后按 FaceDetailer 产出的 `mask` 回贴（仅影响重绘区域）

## 依赖
- `ComfyUI-Impact-Pack`（必须，提供 FaceDetailer）
- `color-matcher`（必须，提供颜色匹配算法）

## 安装
1. 将本目录放入：`ComfyUI/custom_nodes/FaceDetailer_With_Color_Match`
2. 安装依赖（在 ComfyUI Python 环境中）：
   - `pip install -r custom_nodes/FaceDetailer_With_Color_Match/requirements.txt`
3. 重启 ComfyUI

## 节点名
- `FaceDetailer + Color Match`
- 内部类名：`FaceDetailerColorMatch`

## 新增参数
在原 FaceDetailer 参数基础上，新增：
- `apply_color_match`：总开关
- `color_match_method`：`mkl / hm / reinhard / mvgd / hm-mvgd-hm / hm-mkl-hm`
- `color_match_strength`：强度
- `color_match_multithread`：批处理多线程

## 行为说明
- `apply_color_match = disabled`：行为与原 FaceDetailer 保持一致
- `apply_color_match = enabled`：对 FaceDetailer 输出图执行 Color Match（参考图=原图），并按 `mask` 回贴到重绘区域
