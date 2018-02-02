#!/usr/bin/env bash

# skgarden experiments
python skgarden_experiments.py  --input /home/tvas/data/delve  --repeats 10 --window 1000 --njobs 4 --save-predictions   --verbose 1  --output /home/tvas/output/uncertain-trees/delve/MondrianForest

# gnu parallel experiments
parallel -q -j2 java -cp "${MOA_JAR}:/lhome/tvas/repos/moa/moa/target/dependency-jars/*" -javaagent:/lhome/tvas/repos/moa/moa/target/dependency-jars/sizeofag-1.0.0.jar moa.DoTask " EvaluatePrequentialRegression -e (IntervalRegressionPerformanceEvaluator -w 10000) -l (meta.OoBConformalApproximate -i 1000 -a 0.9 -m 100 -d 1 -s 10 -j 4) -s (ArffFileStream -f /lhome/tvas/data/airlines-orig/one-hot/2M/plane_2M_train_{}.arff) -f 10000 -d /lhome/tvas/output/uncertain-trees/airlines-orig/2M/OoBConformalApproximate/plane_2M_train_{}.csv -o  /lhome/tvas/output/uncertain-trees/airlines-orig/2M/OoBConformalApproximate/plane_2M_train_{}.pred" ::: {0..1}

# small-mid parallel for zb
parallel --eta --progress -q -j 8 java -cp "${MOA_JAR}:/lhome/tvas/repos/moa/moa/target/dependency-jars/*" -javaagent:/lhome/tvas/repos/moa/moa/target/dependency-jars/sizeofag-1.0.0.jar moa.DoTask " EvaluatePrequentialRegression -e (IntervalRegressionPerformanceEvaluator -w 1000) -l (meta.{1} -a 0.9 -m 100 -d 1 -s 10 -j 2) -s (ArffFileStream -f /lhome/tvas/data/small-mid/{2}.arff) -f 1000 -d /lhome/tvas/output/uncertain-trees/small-mid/moa-pred/{1}/{2}_{3}.csv -o  /lhome/tvas/output/uncertain-trees/small-mid/moa-pred/{1}/{2}_{3}.pred" ::: OnlineQRF OoBConformalApproximate OoBConformalRegressor ::: 2dplanes calHousing fried mv qsar-chembl240 abalone cpu_act house_16H newsPopularity qsar-chembl253 ailerons electricity_prices house_8L puma32H sulfur bank32nh elevators kin8nm puma8NH yprop_4_1 ::: {0..9}

# moa experiments
python moa_experiments.py  --moajar $MOA_JAR --input /home/tvas/data/airlines-orig/one-hot/10k  --repeats 4 --window 1000 --njobs 2 --meta OoBConformalApproximate --save-predictions --learner-threads 2 --verbose 1 --measure-model-size --output /home/tvas/Dropbox/SICS/uncertain-trees/results/output/airline-orig/test/OoBConformalApproximate

# parameter sweep for algorithm
./parameter_sweep.py --command "python moa_experiments.py  --moajar $MOA_JAR --input /home/tvas/data/moa-regression/large  --repeats 2 --window 1000 --njobs 2 --max-calibration-instances 1000 --save-predictions --learner-threads 1 --verbose 1" --output-prefix /home/tvas/output/uncertain-trees/moa-regression/large --sweep-argument meta --argument-list  OoBConformalApproximate OoBConformalRegressor

# parameter sweep for num calibration instances
./parameter_sweep.py --command "python moa_experiments.py  --moajar $MOA_JAR --input /lhome/tvas/data/moa-regression/  --repeats 10 --window 1000 --njobs 2 --meta meta.OoBConformalRegressor --save-predictions --learner-threads 2 --overwrite --verbose 1" --sweep-argument max-calibration-instances --output-prefix /lhome/tvas/output/uncertain-trees/moa-regression/onlinecp-max-cal-instances --argument-range 100 1001 100

# figures for parameter sweeps
parallel -q -j2 python generate_figures.py --input /home/tvas/Dropbox/SICS/uncertain-trees/results/output/moa-regression/onlineqrf-num-bins/{} --output /home/tvas/Dropbox/SICS/uncertain-trees/results/figures/num-bins/{} --create-tables ::: {10..90}

# nested parameter sweep for algorithm and confidence
./parameter_sweep.py --command "python moa_experiments.py  --moajar $MOA_JAR --input /home/tvas/data/small-mid/  --repeats 1 --window 1000 --njobs 4 --save-predictions --learner-threads 1 --overwrite --verbose 1" --sweep-argument meta --output-prefix /home/tvas/output/uncertain-trees/small-mid/moa-confidence --argument-list OnlineQRF  OoBConformalApproximate  OoBConformalRegressor --inner-sweep-argument confidence --argument-list 0.95 0.99