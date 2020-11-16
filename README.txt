Instructions on how to use Distribution() in the simufit namespace:

import simufit as sf
from simufit import DistributionType as dt
x = sf.Distribution()
x.setDistribution(dt.GEOMETRIC)

x.display
... (Example output)
Seed:  None
Range:  None - None
Size:  None
Samples:  []
Continuous:  None
Discrete: None
Distribution:  DistributionType.GEOMETRIC
...

samples = x.Distribution.sample(p=0.5,size=10) # In the future we can directly set the _sample value directly
x.setSamples(samples)
x.fit() 
x.Distribution.MLE(x._samples, use_minimizer=True, x0=0.42)


Example of loading data from CSV

from simufit.dist_generator import run_fitter
run_fitter()

In the file menu, click Import data
Click Browse... button and select testData.csv
Use comma delimiter, skiprows = 1, use columnn = 1
Click import, histogram should load.
In the main window, select Normal distribution. 
You should now be able to use sliders to fit the data.