



# MapReduce实现并行Apriori算法

## 一、   开发环境

1、     Ubuntu20.04、

2、     eclipse+JDK1.8

3、     Hadoop2.6

## 二、Apriori 算法的原理

   如果某个项集是频繁项集，那么它所有的子集也是频繁的。即如果 {0,1} 是频繁的，那么 {0}, {1} 也一定是频繁的。这个原理直观上没有什么用，但是反过来看就有用了，也就是说如果一个项集是非频繁的，那么它的所有超集也是非频繁的。

\* 实现流程

1. 找出所有频繁项集，过程由连接步和剪枝步互相融合，获得最大频繁项集Lk。

* 连接步目的：找出K项集。

（1） 对给定的最小支持度阀值，分别对候选1-项集C1，剔除小于该阀值的项集得到频繁1-项集L1;

（2）L1自身连接产生2项候选集C2，保留C2中满足约束条件的项集得到2项集L2；

（3）L2与L1连接产生候选3-项集C3，保留C3中满足约束条件的项集得到频繁3-项集L3；

（4）循环下去，得到最大频繁项集Lk

* 剪枝步：紧接着连接步，在产生候选项Ck过程中起到减小搜索空间目的。

​     Ck是Lk-1与L1连接产生，根据Apriori算法原理，如果一个项集是非频繁的，那么它的所有超集也是非频繁的。   不满足的项集以及它的超集不会存在Ck中，即剪枝。

2.由频发项集产生强关联规则，由1获得满足最小置信度阀值频繁项集，因此挖掘出了强关联规则 

## 三、程序实现 

1、    首先将此Java程序打包生成JAR包，然后启动Hadoop,并将数据集上传到HDFS中的“/user/hadoop/input”目录下。

2、     使用hadoop jar命令运行jar文件，格式如下：

```
hadoop jar .<jar包路径> <inp_dir> <out_dir> <min_sup (0.0-1.0)> <min_conf (0.0-1.0)> <txns_count> <delimiter> <max_pass> <filterbylift (0|1)>
```

- *inp_dir*：HDFS中输入数据集的路径
- *out_dir*：HDFS中输出目录的路径，该目录将存储所有中间结果和最终结果。
- *min_sup*：进行频繁项目挖掘的最小支持，该值应在0.0-1.0的范围内
- *min_conf*：关联规则挖掘的最小置信度值，该值应在0.0-1.0的范围内
- *txns_count*：输入数据集中的事务总数，值不应为0。
- *delimiter*：在数据集中用于在单个行中分离多个项目的分隔符，使用“      ”将其概括起来，分隔符为空格表示为：” ”
- *max_pass*：运行Apriori算法的最大迭代次数。如果给定此阈值为5，则会找到所有最大大小为5的常见项目集。
- *filterbylift*：值1将按正提升度的百分比过滤所有规则，最终输出仅包含提升度大于 1.0的规则，否则值0将输出所有规则，而与提升度无关。

具体实现如下：

![2020-06-07 08-47-26屏幕截图](https://github.com/LUXUS1/MapReduce-Apriori/blob/master/photo/2020-06-07%2008-47-26%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png?raw=true)

![2020-06-07 08-48-22屏幕截图](https://github.com/LUXUS1/MapReduce-Apriori/blob/master/photo/2020-06-07%2008-48-22%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png?raw=true)
## 四、实验结果

执行完上述命令后，在Eclipse的 HDFS中的文件列表中会出现包含7个文件的output文件夹，如下图所示：

![2020-06-07 08-56-34屏幕截图](https://github.com/LUXUS1/MapReduce-Apriori/blob/master/photo/2020-06-07%2008-56-34%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)
其中output-pass-k文件夹中为频繁k-项集 ，All-frequent-iteamsets文件夹中为所有频繁项集，rule-mining-out文件夹为所有强关联规则， final-output文件夹为最终输出文件 。
