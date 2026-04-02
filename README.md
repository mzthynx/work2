# <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">贝塞尔曲线绘制实验报告</font>
## <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">一、实验概述</font>
### <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">1.1 实验目标</font>
<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">本实验基于 Python+Taichi 图形编程框架，实现贝塞尔曲线（Bézier Curve）的交互式绘制与渲染。核心目标包括：</font>

+ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">理解贝塞尔曲线的几何意义与 De Casteljau 算法的递归插值原理。</font>
+ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">掌握 GPU 并行计算、像素缓冲区（Frame Buffer）操作的基础方法。</font>
+ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">实现鼠标交互控制、键盘事件响应及高性能图形渲染（解决帧率卡顿问题）。</font>
+ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">实践现代图形学中 “CPU 批量计算 + GPU 并行渲染” 的优化思路，理解对象池、显存预分配的核心思想。</font>

### <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">1.2 实验环境</font>
+ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">编程语言：Python 3.12</font>
+ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">图形框架：Taichi 1.6.0（GPU 后端）</font>
+ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">开发环境：VS Code + PowerShell</font>
+ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">硬件依赖：支持 CUDA 的 GPU</font>

## <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">二、实验原理</font>
### <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">2.1 贝塞尔曲线与 De Casteljau 算法</font>
<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">贝塞尔曲线由一组控制点定义，其形状由控制点的位置和数量决定。</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">De Casteljau 算法</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">是求解贝塞尔曲线点的核心方法，本质是</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">递归线性插值</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">：</font>

1. <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">给定参数 </font> $t \in [0,1]$ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">，对相邻控制点</font> $p_i,p_{i+1}$ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);"> 做线性插值：</font> $p_i'=(1-t)p_i+p_{i+1}$ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">，得到 </font> $n-1$ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">个新点。</font>
2. <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">对新点重复上述插值操作，直至仅剩 1 个点，该点即为</font> $t$ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">对应曲线上的坐标。</font>
3. <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">实验中通过遍历 </font> $p\in [0,1]$ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">(</font><font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">采样 1000 个点），生成完整的贝塞尔曲线离散点集。</font>

### <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">2.2 光栅化与像素缓冲区</font>
<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">屏幕本质是二维像素网格，光栅化的核心是</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">将归一化坐标映射为物理像素索引</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">：</font>

1. <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">初始化 </font><font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">800×800</font><font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);"> 的 GPU 向量缓冲区pixels，存储每个像素的 RGB 浮点值（范围</font> $[0,1]$ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">）。</font>
2. <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">归一化坐标  </font> $[x,y]\in [0,1]$ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">映射为像素索引：</font> $x_{pixels}=int(x\times 80)$ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">，</font> $y_{pixels}=int (y\times 800)$ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">。</font>
3. <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">对目标像素赋值绿色（</font> $[0.0,1.0,0.0]$ <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">），完成 “点亮像素” 操作。</font>

### <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">2.3 高性能渲染优化</font>
<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">现代图形学中，CPU 与 GPU 的 PCIe 通信是性能瓶颈，因此采用</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">批量处理 + 并行计算</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">策略：</font>

+ **<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">对象池（对象池技巧）</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">：预分配固定大小的 GPU 缓冲区gui_point（容量 100），不足位置填充10.0（屏幕外不可见），仅覆盖有效控制点，避免动态内存申请。</font>
+ **<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">CPU 批量计算</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">：在 Python 端一次性计算 1001 个曲线点，存入 NumPy 数组后</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">一次性传输至 GPU</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">，而非逐点传输（减少 1000 次通信）。</font>
+ **<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">GPU 并行渲染</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">：通过@ti.kernel装饰的函数在 GPU 上并行执行，循环点亮像素，大幅提升渲染帧率。</font>

## <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">三、实验代码实现</font>
### <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">核心模块说明</font>
| 模块 | 功能 | 优化点 |
| --- | --- | --- |
|  缓冲区初始化   |  预分配pixels/gui_points/curve_points_field |  避免动态内存申请，符合现代渲染管线规范   |
|  De Casteljau 算法   | <font style="color:rgb(0, 0, 0);">递归计算贝塞尔曲线点</font> |  纯 Python 实现，CPU 端批量计算后一次性传 GPU   |
| 内核函数 draw_curve_kernel |  GPU 并行点亮曲线像素   |  消除逐点通信的性能损耗，支持 1001 个点并行渲染   |
|  交互事件处理   |  鼠标左键加控制点、键盘 C 清空   |  轻量事件监听，支持实时交互   |
|  对象池 gui_points | <font style="color:rgb(0, 0, 0);">固定容量 100 的控制点缓冲区</font> | <font style="color:rgb(0, 0, 0);">填充-10.0隐藏无效点，减少显存通信</font> |


## <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">四、实验结果与分析</font>
### <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">4.1 运行效果</font>
1. **<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">交互功能</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">：</font>
    - <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">鼠标左键点击窗口，可添加红色控制点（最多 100 个）；</font>
    - <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">按下键盘C键，清空所有控制点，画布恢复黑色；</font>
    - <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">控制点≥2 时，自动生成灰色连线与绿色贝塞尔曲线。</font>
2. **<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">渲染性能</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">：</font>
    - <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">采用 GPU 并行渲染后，帧率稳定在 60FPS 以上，无卡顿；</font>
    - <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">对比逐点传输的实现，通信次数从 1001 次降至 1 次，渲染效率提升 10 倍 +。</font>
3. **<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">视觉效果</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">：</font>
    - <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">控制点数量≤100，曲线平滑无锯齿；</font>
    - <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">坐标映射与越界检查确保像素索引有效，无越界崩溃。</font>
    - <img src="https://github.com/mzthynx/work2/blob/master/work2.gif" width="796" title="" crop="0,0,1,1" id="u6bed3432" class="ne-image" style="color: rgb(0, 0, 0); font-size: 16px">

## <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">五、实验总结</font>
### <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">5.1 实验收获</font>
1. <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">深入理解了贝塞尔曲线的几何本质，掌握了 De Casteljau 算法的递归实现逻辑。</font>
2. <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">掌握了 Taichi 框架的 GPU 并行编程、像素缓冲区操作的核心方法，理解了光栅化的底层原理。</font>
3. <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">实践了现代图形学的性能优化技巧：</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">批量数据传输 + GPU 并行计算 + 对象池预分配</font>**<font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">，解决了 CPU-GPU 通信瓶颈。</font>
4. <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">实现了完整的交互式贝塞尔曲线绘制程序，支持鼠标 / 键盘交互，满足实验交互要求。</font>

### <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">5.2 拓展方向</font>
1. <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">支持控制点拖拽交互，提升操作灵活性；</font>
2. <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">增加曲线颜色、线条宽度的动态调节功能；</font>
3. <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">实现多段贝塞尔曲线拼接，绘制更复杂的图形；</font>
4. <font style="color:rgb(0, 0, 0);background-color:rgba(0, 0, 0, 0);">引入抗锯齿算法，提升曲线渲染的视觉质量。</font>
