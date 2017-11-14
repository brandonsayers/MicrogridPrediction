# Versions
* Tensorflow v1.1

# Power-Prediction
Produce a temporal prediction of overall substation power generation & consumption.


## Requirements
  * Perform literature review
  * Be able to predict the next hour's consumption/generation with >90% accuracy
  * Be able to predict up to 12 hours from current period

## Input
  * Missing Data? Use a polynomial fit of other points to fill it in
  * Submodels for discrete power sources. (e.g. unique submodel for three different types of solar panels)

## Output
Report for each substation:
  * Generation (single number in KW/h)
  * Consumption (single number in KW/h)
  * Accuracy
  * Error
  * Deviation and other statistics that we may find useful for validation

## Data 
 * Freq of 10 m, for one year 
 * wind direction & amplitude (No direct measurement from wind turbines) 
 * diesel generator output can be considered the demand, little buffer
 * temperature of environment (affects power generation from solar and power consumption from populace)
 
 
 ### Consider:
   * school days (consumes more power than usual) 
   * build dynamically so that more sources of energy can be added to model 
 
 
## Tips
  * Pull any additional data we may need
  * Angle of the sun
  * Utilize momentum in calculations (SMA for the previous hours leading up to the estimation time)

# Credits
* For lstm model: https://github.com/RobRomijnders/LSTM_tsc

