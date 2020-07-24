package com;


import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.LinkedHashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import com.rules.*;
import com.utils.AprioriAlgorithm;
import com.utils.Utilities;

// 使用Hadoop中MapReduce实现的apriori算法程序 

public class DataDrive extends Configured implements Tool {
	
	private final static String USAGE = "USAGE %s: <input dir path> "
			+ "<output dir path> <min. support> <min. confidence> "
			+ "<transaction count> <transaction delimiter> "
			+ "<max no. of passes> "+ "<enable/disable filter value>\n";
	private static String defFS; // 默认HDFS的值
	private static String inputDir; // HDFS中的输入路径
	private static String outputDir; // HDFS中的输出路径
	private static int txnCount; // 输入数据集中的事务总数
	private static double minSupport; //最小支持度
	private static double minConfidence; //最小置信度
	
	// 用于提取输入数据集中的所有事务的每个项的分隔符
	private static String delimiter;
	private static int maxPass; // Apriori算法的最大迭代次数
	private static boolean liftFilter; // 筛选关联规则
	private static String pathToSavedState; // 临时状态路径
	
	private AprioriAlgorithm apriori = new AprioriAlgorithm();
	private Utilities util = new Utilities();
	
	public static void main(String[] args) throws Exception {
		int exitCode = ToolRunner.run(new DataDrive(), args);
		System.exit(exitCode);
	}

	@Override
	public int run(String[] args) throws Exception {
		if(args.length < 8) {
			
			System.err.printf("Invalid arguments!\n"+USAGE, getClass().getName());
			ToolRunner.printGenericCommandUsage(System.err);
			return 1;
		}
		// 配置保存AprioriAlgorithm的对象状态的路径
		String pwd = Paths.get(".").toAbsolutePath().normalize().toString(); // 当前工作目录
		pathToSavedState = pwd + "/tmp";
		new File(pathToSavedState).mkdir();
		pathToSavedState = pathToSavedState + "/apriori_saved_state.ser";
		new File(pathToSavedState).createNewFile();
		
		// 存储通过命令行接收的参数
		inputDir = args[0];
		outputDir = args[1];
		minSupport = Double.parseDouble(args[2]);
		minConfidence = Double.parseDouble(args[3]);
		txnCount = Integer.parseInt(args[4]);
		delimiter = args[5];
		maxPass = Integer.parseInt(args[6]);
		liftFilter = (args[7].equals("1")) ? true : false;
		
		int minSupportCount = (int)Math.ceil(minSupport * txnCount); // 根据最小支持度阈值支持计算绝对支持
		
		Configuration conf = new Configuration();
		defFS = conf.get("fs.defaultFS");
		apriori.setMinSupportCount(minSupportCount);
		apriori.setMaxPass(maxPass);
		
		// 开始
		jobFrequentItemsetMining(minSupportCount);
		
		// 向HDFS写入找到的所有频繁项集的列表
		String filePath = defFS + outputDir + "/all-frequent-itemsets/freq-list";
		LinkedHashMap<String, Integer> map = new LinkedHashMap<String, Integer>();
		// 按支持计数的降序排列项集
		apriori.getFrequentItemsets().entrySet().stream().sorted((e1, e2) -> (-1) *
				e1.getValue().compareTo(e2.getValue())).forEachOrdered(e -> map.put(e.getKey(), e.getValue()));
		util.addFileToHDFS(conf, filePath, map);
		
		jobAssociationRuleMining();
		jobAssociationRuleAggregation();
		// 结束
		return 0;
	}
	
	// 频繁项集挖掘
	
	private void jobFrequentItemsetMining(int minSupportCount) throws IOException, ClassNotFoundException, InterruptedException {
		String hdfsInputPath = defFS + inputDir;
		String hdfsOutputPath = defFS + outputDir + "/output-pass-";
		boolean success;
		while(!apriori.hasConverged()) {
			int currentPass = apriori.getCurrentPass();
			util.serialize(pathToSavedState, apriori);
			Configuration config = new Configuration();
			config.setInt("APRIORI_PASS", currentPass);
			config.set("DELIMITER", delimiter);
			config.setInt("MIN_SUPPORT_COUNT", minSupportCount);
			config.set("SAVED_STATE_PATH", pathToSavedState);
			config.setBoolean("mapreduce.map.output.compress", true); // 压缩Mapper的输出
			config.setBoolean("mapreduce.output.fileoutputformat.compress", false); // Reducer 输出未压缩 
			Job job = Job.getInstance(config, "Apriori Pass "+currentPass);
			job.setJarByClass(DataDrive.class);
			job.setMapperClass(AprioriPassKMap.class);
			job.setCombinerClass(AprioriPassKCombiner.class);
			job.setReducerClass(AprioriPassKReduce.class);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(IntWritable.class);
			FileInputFormat.addInputPath(job, new Path(hdfsInputPath));
			FileOutputFormat.setOutputPath(job, new Path(hdfsOutputPath+currentPass));
			success = job.waitForCompletion(true);
			if(!success)
				throw new IllegalStateException("Job Apriori Pass "+currentPass+" failed!");
			apriori = util.deserialize(pathToSavedState);
			apriori.nextPass();
		}
		util.serialize(pathToSavedState, apriori);
	}
	
	// Job -> 关联规则挖掘从频繁项集列表中找到完整的有效规则集
	
	private void jobAssociationRuleMining() throws IOException, ClassNotFoundException, InterruptedException {
		String hdfsInputPath = defFS + outputDir + "/all-frequent-itemsets/freq-list";
		String hdfsOutputPath = defFS + outputDir + "/rule-mining-output";
		Configuration config = new Configuration();
		config.setDouble("MIN_CONFIDENCE", minConfidence);
		config.setInt("TRANSACTION_COUNT", txnCount);
		config.setBoolean("LIFT_FILTER", liftFilter);
		config.set("SAVED_STATE_PATH", pathToSavedState);
		Job job = Job.getInstance(config, "Association Rule Mining");
		job.setJarByClass(DataDrive.class);
		job.setMapperClass(AssociationRuleMiningMap.class);
		job.setReducerClass(AssociationRuleMiningReduce.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job, new Path(hdfsInputPath));
		FileOutputFormat.setOutputPath(job, new Path(hdfsOutputPath));
		boolean success = job.waitForCompletion(true);
		if(!success)
			throw new IllegalStateException("Job Association Rule Mining failed!");
	}
	
	// Job -> 关联规则聚合以删除任何冗余规则以及最终输出
	
	private void jobAssociationRuleAggregation() throws IOException, ClassNotFoundException, InterruptedException {
		String hdfsInputPath = defFS + outputDir + "/rule-mining-output";
		String hdfsOutputPath = defFS + outputDir + "/final-output";
		Configuration config = new Configuration();
		Job job = Job.getInstance(config, "Association Rule Aggregation");
		job.setJarByClass(DataDrive.class);
		job.setMapperClass(RuleAggregatorMap.class);
		job.setReducerClass(RuleAggregatorReduce.class);
		job.setPartitionerClass(RuleAggregatorPartitioner.class);
		job.setGroupingComparatorClass(RuleAggregatorGroupComparator.class);
		job.setSortComparatorClass(RuleAggregatorSortComparator.class);
		job.setOutputKeyClass(Text.class);
		job.setOutputValueClass(Text.class);
		FileInputFormat.addInputPath(job, new Path(hdfsInputPath));
		FileOutputFormat.setOutputPath(job, new Path(hdfsOutputPath));
		boolean success = job.waitForCompletion(true);
		if(!success)
			throw new IllegalStateException("Job Association Rule Aggregation failed!");
	}
	
}
