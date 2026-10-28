# 可选挑战：云端扩展尺度实验的操作与费用

本次实际计算在本机进行，**没有创建或连接云服务器**。以下是需要时人工操作的示例；
账号、地区、配额、镜像和价格都需要在启动前自行确认。此处的示例不会自动创建资源。

## 先估算费用

示例选择美国东部弗吉尼亚 `us-east-1`、Linux 按需实例 `c7i.8xlarge`。AWS 标明它有
32 vCPU、64 GiB 内存；其 CPU 配置是 16 个物理核心、每核心 2 线程，因此满足“16 个
CPU 核心”这一较严格的理解。[AWS 规格](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/cpu-options-supported-instances-values.html)

2026-10-28 检索时，第三方价格表列出该型号在此地区约 **$1.428/小时**；仅作为估算，
启动前应到 [AWS 定价计算器](https://calculator.aws/)核对实际账单。
[价格表](https://www.doit.com/compute/compute/aws/us-east-1/c7i.4xlarge)
假设机器运行 2 小时，计算费约 $2.856；再假设 20 GiB gp3 盘保留 2 小时，按 $0.08/
GiB·月约 $0.004；一个公网 IPv4 按 $0.005/小时约 $0.01。合计约 **$2.87**，未包含
税、数据传出、额外性能配置或可能保留的快照。两小时是预算假设，不是已测得的云端
训练时间。[EBS 价格示例](https://docs.aws.amazon.com/us_en/emr/latest/ManagementGuide/emr-plan-storage-compare-volume-types.html)；
[公网 IPv4 定价](https://aws.amazon.com/vpc/pricing/)。

## 启动与连接

1. 在 AWS EC2 控制台选择 `us-east-1`，确认按需 vCPU 配额允许 32 vCPU。创建一个
   Linux 用 ED25519 密钥对并保管好 `.pem` 私钥；选择 Amazon Linux 2023 或兼容的
   64 位 Linux 镜像、`c7i.8xlarge` 和足够的 gp3 系统盘。
2. 给实例分配公网地址；安全组仅允许**自己的公网 IP**连接 TCP 22。确认根磁盘的
   “终止时删除”设置，再查看费用和配置后启动。
3. 在本机终端按照控制台显示的用户名和地址连接；下面的主机名与密钥名是占位符：

   ```sh
   chmod 400 /path/to/week8-key.pem
   ssh -i /path/to/week8-key.pem ec2-user@YOUR_EC2_HOST
   ```

   AMI 不同时用户名也可能不同。参见 [AWS 创建密钥和连接教程](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/tutorial-launch-a-test-ec2-instance.html)。
4. 在远端安装 Git、Python 3.14 和 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
   如需运行本次本地版本，可将 `week8/` 源码、配置、`uv.lock`、
   `data/input.txt`、`speedrun/input.txt` 和 `artifacts/scaling-prediction.json` 复制到
   远端的新工作目录。不要复制 `.venv/`，也不要把已有的 256/384 结果复制到新的
   盲测目录。远端运行 `uv sync --locked`。

## 保留事前预测顺序

在一个没有 256/384 训练结果的全新远端 `week8/` 中，把本次冻结的原始拟合预测
文件放在 `artifacts/scaling-prediction.json`，建立只含原始拟合数据的记录：

```sh
python - <<'PY'
import json
from pathlib import Path
prediction = json.loads(Path('artifacts/scaling-prediction.json').read_text())
Path('data').mkdir(exist_ok=True)
Path('data/cpu-measurements.json').write_text(json.dumps({'part3b': {'prediction': prediction}}, indent=2))
PY
uv run python cpu/scaling.py predict-extra cpu/configs/scaling-b.json --width 256
uv run python cpu/scaling.py predict-extra cpu/configs/scaling-b.json --width 384
```

先把两个预测写成独立快照，并核对时间，之后才运行训练：

```sh
python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path('data/cpu-measurements.json').read_text())['part3b']['extra_heldout']
with Path('artifacts/scaling-extra-predictions.json').open('x') as f:
    json.dump(p, f, indent=2)
PY
```

```sh
uv run python cpu/scaling.py train-extra cpu/configs/scaling-b.json --width 256 --jobs 2
uv run python cpu/scaling.py train-extra cpu/configs/scaling-b.json --width 384 --jobs 2
uv run python cpu/scaling.py compare-extra cpu/configs/scaling-b.json --width 256
uv run python cpu/scaling.py compare-extra cpu/configs/scaling-b.json --width 384
```

例如在本机先创建一个新的接收目录，然后用 `scp -i /path/to/week8-key.pem -r
ec2-user@YOUR_EC2_HOST:REMOTE_WEEK8/artifacts/scaling LOCAL_NEW_DIRECTORY/` 复制运行
目录；另行复制 `data/cpu-measurements.json` 和预测快照。检查文件完整后，在 EC2
控制台选择对应实例并 **Terminate**，核对状态最终变为 `terminated`；检查是否仍有
需要计费的 EBS 卷、快照或保留的弹性 IP。不要在确认数据已经复制之前终止实例。
