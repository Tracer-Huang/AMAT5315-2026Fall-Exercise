# Silent two-minute screen demonstration

Purpose: show a real fresh-clone physics check reaching PASS, then show the public heating trajectory near T=0.2 and T=1.0. The student explicitly requested a silent screen demonstration, so no narration is added. The delivered demonstration uses actual chronological browser captures and real fresh-shell output displayed in a read-only panel. Browser capture waits are uniformly time-compressed to 110 seconds; it is not labeled a native desktop screencast.

## Screen sequence

1. Start in a fresh clone's week2 directory. Show and run:

   ```sh
   make reproduce
   cargo run --manifest-path md/Cargo.toml --release -- check artifacts
   ```

2. Once PASS is visible, open https://tracer-huang.github.io/AMAT5315-2026Fall-Exercise/ . Show the header identifying 400 atoms and 200 frames.
3. Pause at frame 0 (current T about 0.204), then near frame 160 (current T about 1.002). Explain the g(r) peaks at each state.
4. Optionally show frame 199: current T about 1.197 and long-range contrast about 0.096, below the cold start's 0.333.

The viewer's speed histogram averages 20 frames; its g(r) averages up to 10 frames. The current-temperature history label is the relevant instantaneous T for the pause points.

## Reading notes (not an audio track)

这是 Week 2 的二维 Lennard-Jones 分子动力学程序。我从全新克隆的仓库重新生成轨迹，再从保存的位置和速度重新计算能量、温度和速率分布。这里三个检查都在规定界限内，程序显示 PASS。

公开网页展示了四百个原子的两百帧加热轨迹。现在接近零点二的低温，原子主要围绕晶格位置振动。径向分布函数除了第一个近邻峰，在更远的距离仍有多层明显的峰，说明存在晶体的空间有序性。

温度升到约一点零后，原子能够交换邻居，远处的峰明显减弱，曲线逐渐接近一。第一个近邻峰仍然存在，因为原子仍然不能相互重叠。加热结束时，长程结构对比度由约零点三三三降到零点零九六。这些结构变化说明晶体的长程有序性正在消失。加热时总能量增加是预期行为，所以能量守恒检查使用的是前面的未加热默认实验。

## Delivery

Attach the recording file to a GitHub release of Tracer-Huang/AMAT5315-2026Fall-Exercise and place its link beside the Pages URL in README.md. Keep the video itself out of git. Do not claim the recording requirement complete until the actual media has been inspected and uploaded.
