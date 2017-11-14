# Author: Rocco Haro
# Capstone Project: Power prediction for micro grid.
# University of Alaska, Anchorage
# rocco.haro18@gmail.com

# Data processing (& cleaning) for power grid data.

# Resources:
# https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

# to read the data from the above dir
from os import listdir
import os.path

# Custom debug class. Just dynamically prints <varName>, <varValue>
# e.g. if i have x = 1, then display(x) -> <'x'>, <1>
import displayClass as DC

# for converting dates represented as strings to a datetime format
# now we can normalize the date time
from dateutil import parser

# for mapping the month to an integer
import calendar

# for stripping date time
from datetime import datetime, date

class dataProcessor:
    def __init__(self, *args):
        """ args: < file path of dir> ,
        """
        self.args = args
        self.dataFilePath = self.args[0]
        self.declaredFiles = self.getFilesPaths()
        self.debug = True
        self.rawDataSet = []
        self.matchedData = []
        # Crucial !!!!!!!!!!!!!!!!!!!!!
        self.baseYear = 2010

        self.monthMapping = None
        self.initMonthMapping()

    def getFilesPaths(self):
        """ Returns a list of all the data files passed in by the user by
            locating the -df flag, and storing the next argum. in the list.
        """
        nextIsFileName = False
        filePaths = []
        for arg in self.args[1]:
            if (nextIsFileName):
                filePaths.append(arg)
                nextIsFileName = False
            else:
                if (arg == '-df'):
                    nextIsFileName = True
        return filePaths

    def initMonthMapping(self):
        self.monthMapping =  {name: num for num, name in enumerate(calendar.month_abbr) if num}

    def help(self):
        print("==== Help")
        print(""" The first file must be formatted as the following: <date&time> <output/label
        \n        The remanding files are features that are mapped to the labels by the dataProcessor class""")


    def readData(self):
        """ THE FIRST FILE READ IN IS CONSIDERED THE LABEL OUTPUT.
            THE OUTPUT FOR THE TIME-SERIES DATA MUST NATURALLY BE IN A SEPERATE FILE
        """

        def processInstance(potentialInstance, isPowerstation):
            """ CRUCIAL. Data must be passed in with
                date & time first. Additionally, for first prototype,
                no values with NaN or N will be accepted.
                Possible acceptance states:
                (1) <year:month:day>, < hrs:min:sec>, < ... actual info ... >
                (2) <year:month:day hrs:min:sec>, < ... actual info ... >

            """
            def cumulativeDaysUptToThisMonth(year, month):
                daysAcc = 0
                #print("cumulativeDaysUptToThisMonth")
                try:
                    x = int(month)
                    for key, value in self.monthMapping.items():
                        if value < x:
                            # returns tuple of start day of week, and number of days in month
                            daysAcc+= calendar.monthrange(int(year), value)[1]
                except:
                    # month is represented as 3 char string
                #    print("Month value: ", self.monthMapping[month])
                    for key, value in self.monthMapping.items():
                        if value < self.monthMapping[month]:
                            # returns tuple of start day of week, and number of days in month
                            daysAcc+= calendar.monthrange(int(year), value)[1]
                return daysAcc

            def getTotalTimeInMin(year, month,days, hrs, mins):
                """ Returns total mins that have elapsed since self.baseYear
                """
                def getTotalDays(year, month, days):
                    """ converts all time passed to minutes. subtracted from self.baseYear
                        returns int
                    """
                    daysPassedinMonth = cumulativeDaysUptToThisMonth(year, month)
                    print("daysPassedinMonth: ", daysPassedinMonth)
                    daysFromYear = (int(year) - self.baseYear)*365
                    totalDays = daysPassedinMonth + int(days) + daysFromYear
                    return totalDays

                totalDays = getTotalDays(year, month, days)
                return int( float(totalDays)*24*60 + hrs*60 + mins)


            # TODO optomize by first checking for NaN (the last variable)

            try:
                # checks if there is a null value
                test = float(potentialInstance[-1])

                # TODO handle the null values
            except:
                return False

            # file and data format:
            # "powerstation.csv" , "-sprte", "DD:MMM:YY", "HH:MM:SS",
            if (isPowerstation):
                try:
                    dateInfos = str(potentialInstance[0]).split('-')
                    day = (dateInfos[0])
                    month = dateInfos[1]
                    year = "20"+dateInfos[2]

                    timeInfo = potentialInstance[1].split(':')
                    hrs = float(timeInfo[0])
                    mins = float(timeInfo[1])

                    totalTimeInMin = getTotalTimeInMin(year, month, day, hrs, mins)

                    powerOutput = float(potentialInstance[2])

                    if (self.debug and False):
                        dC = DC.displayFuncs(list(locals().iteritems()))
                        dC.display(dateInfos,timeInfo, totalTimeInMin)
                        #raw_input()
                    processedInstance = [totalTimeInMin, powerOutput]
                    return processedInstance
                except:
                    print("faield for power station data conversion.")
                    print("instance: ", potentialInstance)
                    raw_input()

            # file and data format:
            #"wind_data.csv", "-tgthr" "MM:DD:YYYY HH:MM"
            else:
                try:
                    dateInfos, timeInfo = potentialInstance[0].split(' ')
                    dateInfos = dateInfos.split('/')

                    month = dateInfos[0]
                    day = dateInfos[1]
                    year = dateInfos[2]

                    timeInfo = timeInfo.split(':')
                    hrs = float(timeInfo[0])
                    mins = float(timeInfo[1])

                    totalTimeInMin = getTotalTimeInMin(year, month, day, hrs, mins)

                    processedInstance = [totalTimeInMin]

                    for i in range(1, len(potentialInstance)):
                        processedInstance.append(float(potentialInstance[i]))

                    if (self.debug and totalTimeInMin == 2115790):
                        dC = DC.displayFuncs(list(locals().iteritems()))
                        dC.display(processedInstance, month, day, year, hrs, mins)
                        raw_input()
                    return processedInstance
                except:
                    print("****************************** failed for wind data.")
                    raw_input()
                    return False
        """ Purpose: read from the data folder,
            which should be located above the curr
            working directory.
        """
        basepath = os.path.dirname(__file__)
        folderPath = os.path.abspath(os.path.join(basepath, "..", self.dataFilePath))
        dataFiles = [f for f in listdir(folderPath)
                    if os.path.isfile(os.path.join(folderPath,f))]

        for exstingFile in dataFiles:
            if exstingFile not in self.declaredFiles:
                print("*** existing file is not in the declared files.")
                return exstingFile
        if(self.debug and False):
            # pass all the current variables that have been declared
            dC = DC.displayFuncs(list(locals().iteritems()))
            dC.display(basepath,folderPath, dataFiles)


        for dataFile in self.declaredFiles:
            isPowerstation = dataFile == "powerstation.csv"
            print(dataFile)
            with open(folderPath+"/"+dataFile, 'r') as fileInstance:
                for line in fileInstance:
                    instanceInfos = [x.rstrip() for x in line.split(',')]

                    proccessedInfo = processInstance(instanceInfos, isPowerstation)

                    # processed Info will return False if there is missing information
                    # later missing information may be substituted for interpolated data
                    if (proccessedInfo):
                        self.rawDataSet.append(proccessedInfo)


                    if (self.debug and False): # Exra bool so i can toggle printing
                        dC.updateVarInfolist(locals().iteritems())
                    #    raw_input()
                        dC.display(instanceInfos, proccessedInfo)

        return True

    def suicide(self):
        print("Killing.")
        quit()

    def extractValidData(self):
        """ returns a set of instances whose features are not null
            loop through.
            It's given that the first range is the range of labels.
            Therefore build a list consisting only of that range,
            and then start mapping the rest of the data into the new list

            validInstance format: dict(Key: timestamp, Value: [label, [feats, ...])
        """
        print("=== extractValidData")
        firstInstance = self.rawDataSet[0]
        firstTimeStamp = firstInstance[0] - 10
        currTimeStamp = firstTimeStamp
        validInstances = dict()
        doneInitializing = False
        print("First time stamp: ", firstTimeStamp)
        # extract the range for the labels
        for instance in self.rawDataSet:
            instanceTimeStamp = instance[0]
            # only go forward in time
            if (instanceTimeStamp > currTimeStamp ):
                # append timestamp and outputlabel
                label = instance[1]
                # <timestamp> <outputlabel> <list for features>
                validInstances[instanceTimeStamp] = [label, []]
                currTimeStamp = instanceTimeStamp # 2607840
            else:
                print("Max date: ", currTimeStamp)
                #raw_input()
                break

            #dC = DC.displayFuncs(list(locals().iteritems()))
        for instance in self.rawDataSet:
            # ensuring we arent copying over label again
            currTimeStamp = instance[0]
            if currTimeStamp in validInstances.keys():
                if instance[1] not in validInstances[currTimeStamp]:
                    #raw_input()
                    validInstances[currTimeStamp][1]+=instance[1:]
                    x = validInstances[currTimeStamp]
                #    dC = DC.displayFuncs(list(locals().iteritems()))
                #    dC.display(x, instance, currTimeStamp)
                    #raw_input()
                    if len(x[1]) > 3:
                        raw_input()
        #    raw_input()
            #dC.display(instance)
        #print(validInstances)

    def run(self):
        """ * starts the script.
            * debug mode prints statements of what
                the script is performing.
        """
        self.help()
        if (self.debug):
            print("==== running")
            processStatus = self.readData() # returns a file name if it errors out
            print("procststs", processStatus)
            if (processStatus):

                #print("*** "+processStatus)
                self.extractValidData()
                self.suicide()
            else:
                print("*** Process status returned False.")

        # else:
        #     print("=== running")
        #     processStatus = self.readData() # returns a file name if it errors out
        #     if (processStatus):
        #         print("*** A file was found that you did not provide date format info for: ")
        #         print("*** "+processStatus)
        #         self.extractValidData()
        #         self.suicide()

if __name__ == "__main__":
    #TODO have user input from cmd line for the files.

    # TODO the gui should be in another file
    # also give instructions on how to input the data by calling help()

    # add flags to identify whether the data and hours are together or separate
    # preprocess so that when -sprte it splits the date and time while
    # -tgthr keeps them together
    fileArgumentsFromUser = ["-df", "powerstation.csv" , "-sprte", "DD:MMM:YY", "HH:MM:SS", "-df" , "wind_data.csv", "-tgthr" "MM:DD:YYYY", "HH:MM"]
    dp = dataProcessor("data", fileArgumentsFromUser)
    dp.run()
