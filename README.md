# Simufit Guide

## Example Distribution:

### Import simufit and Distribution Types (i.e. Bernoulli, Uniform, Geometric, etc...):
```
import simufit as sf
from simufit import DistributionType as dt
```
### Create an instance of the Distribution class and set the distribution type:
```
x = sf.Distribution()
x.setDistribution(dt.UNIFORM)
```
### Generate Sample Data:
```
x.generateSamples(a=10,b=20,size=200,seed=123)
```

### Display the distribution object information:
```

x.display()
..........
Seed:  123
Range:  10.026880645743207 - 19.953584820340176
Size:  200
Samples:  [16.96469186 12.86139335 12.26851454 15.51314769 17.1946897  14.2310646
 19.80764198 16.84829739 14.80931901 13.92117518 13.43178016 17.29049707
 14.38572245 10.59677897 13.98044255 17.37995406 11.8249173  11.75451756
 15.31551374 15.31827587 16.34400959 18.49431794 17.24455325 16.11023511
 17.22443383 13.22958914 13.61788656 12.28263231 12.93714046 16.30976124
 10.9210494  14.33701173 14.30862763 14.93685098 14.2583029  13.12261223
 14.26351307 18.93389163 19.44160018 15.01836676 16.23952952 11.15618395
 13.17285482 14.14826212 18.66309158 12.50455365 14.83034264 19.85559786
 15.19485119 16.12894526 11.20628666 18.26340801 16.03060128 15.45068006
 13.42763834 13.04120789 14.17022211 16.81300766 18.75456842 15.10422337
 16.69313783 15.85936553 16.24903502 16.74689051 18.42342438 10.83194988
 17.63682841 12.43666375 11.94222961 15.72456957 10.95712517 18.85326826
 16.27248972 17.23416358 10.16129207 15.94431879 15.56785192 11.58959644
 11.53070515 16.95529529 13.18766426 16.91970296 15.5438325  13.88950574
 19.2513249  18.41669997 13.57397567 10.43591464 13.04768073 13.98185682
 17.0495883  19.95358482 13.55914866 17.62547814 15.93176917 16.91701799
 11.51127452 13.98876293 12.40855898 13.43456014 15.13128154 16.6662455
 11.05908485 11.30894951 13.21980606 16.61564337 18.46506225 15.53257345
 18.54452488 13.84837811 13.16787897 13.54264676 11.71081829 18.29112635
 13.38670846 15.52370075 15.78551468 15.21533059 10.02688065 19.88345419
 19.05341576 12.07635861 12.92489413 15.20010153 19.01911373 19.83630885
 12.57542064 15.64359043 18.06968684 13.94370054 17.31073036 11.61069014
 16.00698568 18.65864458 19.83521609 10.7936579  14.28347275 12.0454286
 14.50636491 15.47763573 10.9332671  12.96860775 19.2758424  15.69003731
 14.57411998 17.53525991 17.41862152 10.48579033 17.08697395 18.39243348
 11.65937884 17.80997938 12.86536617 13.06469753 16.65261465 11.11392172
 16.64872449 18.87856793 16.96311268 14.40327877 14.38214384 17.65096095
 15.65642001 10.84904163 15.82671088 18.14843703 13.37066383 19.2757658
 17.50717    15.74063825 17.51643989 10.79148961 18.59389076 18.21504113
 19.0987166  11.28631198 10.81780087 11.38415573 13.9937871  14.24306861
 15.62218379 11.2224355  12.01399501 18.11644348 14.67987574 18.07938209
 10.07426379 15.51592726 19.31932148 15.82175459 12.06095727 17.17757562
 13.7898585  16.68383947 10.29319723 16.35900359 10.32197935 17.44780655
 14.72913002 11.21754355]
Measure Type:  CONTINUOUS
Distribution:  UNIFORM
..........
```
### Fit the Data and Get the Maximum Likelihood Estimate:
```
x.fit()
x.Distribution.MLE(x._samples, use_minimizer=True, x0=0.42)
```

### Identify Unknown Distribution
To run multiple tests (i.e. MLE, GOF, etc..) to identify the distribution, use identifyDistribution with the required parameters for the MLE of a given distribution type each one will be executed. If a parameter is missing, that distribution will be skipped.

The output will indicate the success of the evalaution steps or any issues encountered for each distribution.
```
x.identifyDistribution()

x.identifyDistribution(a0=9,b0=27, use_minimizer=True,p0=0.5,mean0=17.5,var0=4,n=1)

Evaluating Geometric
-----------------------------------------------
Starting...
...\simufit\dist_generator.py:537: RuntimeWarning: invalid value encountered in log
  return (n * np.log(p) + np.sum(samples - 1) * np.log(1 - p)) * -1
Completed
-----------------------------------------------



Evaluating Uniform
-----------------------------------------------
Starting...
Completed
-----------------------------------------------



Evaluating Normal
-----------------------------------------------
Starting...
Completed
-----------------------------------------------



Evaluating Exponential
-----------------------------------------------
Distribution Skipped
-----------------------------------------------



Evaluating Gamma
-----------------------------------------------
Starting...
...\simufit\dist_generator.py:746: RuntimeWarning: invalid value encountered in log
  return ((a - 1) * np.sum(np.log(samples)) - n * scipy.special.gamma(a) - n * a * np.log(b) - (np.sum(samples) / b)) * -1
Warning: Optimizer failed to converge with initial guess 9, 27. Returned None for MLE values. Try another initial guess.
Completed
-----------------------------------------------



Evaluating Bernoulli
-----------------------------------------------
Starting...
...\simufit\dist_generator.py:435: RuntimeWarning: invalid value encountered in log
  return (m * np.log(p) + (n - m) * np.log(1 - p)) * -1
...\simufit\dist_generator.py:435: RuntimeWarning: divide by zero encountered in log
  return (m * np.log(p) + (n - m) * np.log(1 - p)) * -1
Warning: Optimizer failed to converge with initial guess 0.5. Returned None for MLE values. Try another initial guess.
Completed
-----------------------------------------------



Evaluating Binomial
-----------------------------------------------
Starting...
...\simufit\dist_generator.py:479: RuntimeWarning: invalid value encountered in log
  return (np.sum(scipy.special.comb(n, samples)) + np.sum(samples) * np.log(p) + (n * len(samples) - np.sum(samples)) * np.log(1 - p)) * -1
...\simufit\dist_generator.py:479: RuntimeWarning: divide by zero encountered in log
  return (np.sum(scipy.special.comb(n, samples)) + np.sum(samples) * np.log(p) + (n * len(samples) - np.sum(samples)) * np.log(1 - p)) * -1
Warning: Optimizer failed to converge with initial guess 0.5. Returned None for MLE values. Try another initial guess.
Completed
-----------------------------------------------



Evaluating Weibull
-----------------------------------------------
Starting...
...\simufit\dist_generator.py:804: RuntimeWarning: invalid value encountered in log
  return (n * np.log(a) - n * np.log(b) + (a - 1) * np.sum(np.log(samples / b)) - np.sum(np.power((samples / b), a))) * -1
...\simufit\dist_generator.py:804: RuntimeWarning: invalid value encountered in power
  return (n * np.log(a) - n * np.log(b) + (a - 1) * np.sum(np.log(samples / b)) - np.sum(np.power((samples / b), a))) * -1
Warning: Optimizer failed to converge with initial guess 9, 27. Returned None for MLE values. Try another initial guess.
Completed
-----------------------------------------------
```
### Displaying an Evaluation Report
To display the values calculated by each of the identification of Distribution tests, use the printReport() method.

```
x.printReport()

-------------

Distribution Type:  Geometric
MLE:  [0.11176758]
Goodness of Fit:  None
-------------

Distribution Type:  Uniform
MLE:  (-24.83433420862302, 36.50829018537624)
Goodness of Fit:  None
-------------

Distribution Type:  Normal
MLE:  [ 8.94883253 97.37910797]
Goodness of Fit:  None
-------------

Distribution Type:  Exponential
MLE:  Not Performed
Goodness of Fit:  Not Performed
-------------

Distribution Type:  Gamma
MLE:  (None, None)
Goodness of Fit:  None
-------------

Distribution Type:  Bernoulli
MLE:  None
Goodness of Fit:  None
-------------

Distribution Type:  Binomial
MLE:  None
Goodness of Fit:  None
-------------

Distribution Type:  Weibull
MLE:  (None, None)
Goodness of Fit:  None
-------------

```


## Creating a Pip Package:
### Creating the whl
```
python setup.py bdist_wheel
```
### Creating the Requirements.txt File
```
pip wheel -r .\src\requirements.txt
```

## Example of loading data from CSV

### Example 1: 5000 gamma-distributed samples with a=2.3, b=5.4
```
from simufit import Display as dp
dp.run_fitter()
```
- Import the tests/gammaTestData.txt file. Use skip rows=1, use column=1, delimter=semi-colon
- Select Gamma from the Distribution pull-down menu
- Press the auto-fit button.

### Example 2: 10000 geometric-distributed samples with p=0.67
```
from simufit import Display as dp
dp.run_fitter()
```
- Import the tests/geometricTestData.txt file. Use skip rows=1, use column=0, delimter=comma
- Select Geometric from the Distribution pull-down menu
- Press the auto-fit button.

### Example 3: 1000 normally distributed samples with mean=0, variance=1
```
from simufit import Display as dp
dp.run_fitter()
```
- Import the tests/normalTestData.txt file. Use skip rows=0, use column=1, delimter=comma
- Select Normal from the Distribution pull-down menu
- Press the auto-fit button.

### Example 4: 1000 Weibull-distributed samples with a=2, b=3
```
from simufit import Display as dp
dp.run_fitter()
```
- Import the tests/weibullTestData.txt file. Use skip rows=0, use column=0, delimter=comma
- Select Weibull from the Distribution pull-down menu
- Press the auto-fit button.
