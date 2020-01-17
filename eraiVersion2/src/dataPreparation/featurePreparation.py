import os, errno, traceback, sys

from pathlib import Path

traceback_template = '''Traceback (most recent call last):
  File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n''' # Skipping the "actual line" item

def doBasicOperation(dataName,dataFrequency):
    import os,sys,traceback    
    from datetime import datetime, timedelta

    import pandas as pd  
    import numpy as np
    
    from config.environment import getAppConfigData
    from config.environment import setAppConfigData
    
    from utilities.fileFolderManipulations import getJupyterRootDirectory
    from utilities.fileFolderManipulations import getParentFolder
    from utilities.fileFolderManipulations import createFolder
    print ("into method doBasicOperation")

    return_fundamentalFeaturesDf = None

    try:

        # Variable to hold the original source folder path which is calculated from the input relative path of the source folder (relativeDataFolderPath)
        # using various python commands like os.path.abspath and os.path.join
        jupyterNodePath = getJupyterRootDirectory()

        configFilePath = None    

        # holds data from input data file - Truth source, should be usd only for reference and no updates should happen to this variable
        inputRawProcessedDataDF = None    

        autoConfigData = getAppConfigData()        

        preProcessedDataFilePath=autoConfigData[dataName][dataFrequency]['preProcessedDataFilePath']

        # read the raw processed data from csv file
        inputRawProcessedDataDF = pd.read_csv(jupyterNodePath + preProcessedDataFilePath)  

        return_fundamentalFeaturesDf = createFundamentalFeatures(inputRawProcessedDataDF)
        
        print("before return statement of method doBasicOperation ")
   
    except:
        print("Error executing method >>> ")
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print("Unexpected error:", sys.exc_info())
        # print(exc_type, fname, exc_tb.tb_lineno)
        
        # http://docs.python.org/2/library/sys.html#sys.exc_info
        exc_type, exc_value, exc_traceback = sys.exc_info() # most recent (if any) by default
        
        '''
        Reason this _can_ be bad: If an (unhandled) exception happens AFTER this,
        or if we do not delete the labels on (not much) older versions of Py, the
        reference we created can linger.

        traceback.format_exc/print_exc do this very thing, BUT note this creates a
        temp scope within the function.
        '''

        traceback_details = {
                            'filename': exc_traceback.tb_frame.f_code.co_filename,
                            'lineno'  : exc_traceback.tb_lineno,
                            'name'    : exc_traceback.tb_frame.f_code.co_name,
                            'type'    : exc_type.__name__,
                            'message' : traceback.extract_tb(exc_traceback)
                            }
        
        del(exc_type, exc_value, exc_traceback) # So we don't leave our local labels/objects dangling
        # This still isn't "completely safe", though!
        # "Best (recommended) practice: replace all exc_type, exc_value, exc_traceback
        # with sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]
        
        print
        print(traceback.format_exc())
        print
        print(traceback_template % traceback_details)
        print

        #traceback.print_exception()
        raise
    finally:
        return return_fundamentalFeaturesDf

def getAttributeDifferenceBasedFeatures(df):
    import pandas as pd

    return pd.concat([
            (df['open']-df['close']).rename('open_close_diff'),
            (df['open']-df['high']).rename('open_high_diff'),
            (df['open']-df['low']).rename('open_low_diff'),        
            (df['close']-df['high']).rename('close_high_diff'),
            (df['close']-df['low']).rename('close_low_diff'),
            (df['high']-df['low']).rename('high_low_diff')
    ],axis=1)

def getAttributeDifferenceBasedMidPointsFeatures(df):
    import pandas as pd

    return pd.concat([
            ((df['open']+df['close'])/2).rename('open_close_mid'),
            ((df['open']+df['high'])/2).rename('open_high_mid'),
            ((df['open']+df['low'])/2).rename('open_low_mid'),
            ((df['close']+df['high'])/2).rename('close_high_mid'),
            ((df['close']+df['low'])/2).rename('close_low_mid'),
            ((df['high']+df['low'])/2).rename('high_low_mid')
    ],axis=1)

def getTrainableFeaturesListDf(filePath):
    import pandas as pd
    df = None
    try:
        df=pd.read_csv(filePath) 
    except FileNotFoundError:
        df=None
        
    return df
    
def createFundamentalFeatures(rawDf):    
    import pandas as pd

    #initialize the straight forward input features
    df = pd.DataFrame({            
        'open':rawDf['open'],
        'high':rawDf['high'],
        'low':rawDf['low'],
        'close':rawDf['close']    
    })

    print("added INPUT FEATURES >>> 4 count >>> open-high-low-close")    
    
    return df

def _createNewTrainingSetWithFeatureVariations(basicDf,newFeatureDf,featureOfInterest,variation_degree) :
    import pandas as pd
    import numpy as np

    from tqdm import tqdm
    from utilities.pandasTools import suffixColumnsWithLabel

    try:
        # Create and register a new `tqdm` instance with `pandas`
        # (can use tqdm_gui, optional kwargs, etc.)
        tqdm.pandas()


        featureVariants=[[
                            np.exp(suffixColumnsWithLabel(newFeatureDf,'_exp_'+str(iterator))*iterator), 
                            np.exp(suffixColumnsWithLabel(newFeatureDf,'_exp_inv_'+str(iterator))*iterator*-1),
                            np.power(suffixColumnsWithLabel(newFeatureDf,'_pow_'+str(iterator)),iterator),
                            np.power(suffixColumnsWithLabel(newFeatureDf,'_pow_inv_'+str(iterator)).astype(float),iterator*-1)
        ] for iterator in range(1,variation_degree+1)]
        
        segmentCount,rowCount, colCount = len(featureVariants),len(featureVariants[0]),len(featureVariants[0][0])

        cummulativeListOfFeatures = np.empty(segmentCount, dtype=list)
        
        for segmentItr in range(0,segmentCount-1):
            cummulativeListOfFeatures[segmentItr]=pd.DataFrame([])
            for rowItr in range(0,rowCount-1): 
                cummulativeListOfFeatures[segmentItr] = pd.concat([cummulativeListOfFeatures[segmentItr],featureVariants[segmentItr][rowItr]],axis=1)
        
        cummulativeListOfFeatures=pd.concat(cummulativeListOfFeatures,axis=1)
        
        newTrainingSetDf= pd.concat([basicDf,newFeatureDf,cummulativeListOfFeatures],axis=1)
       
        return newTrainingSetDf
    except:
        print("Error executing method >>> ")
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print("Unexpected error:", sys.exc_info())
        # print(exc_type, fname, exc_tb.tb_lineno)
        
        # http://docs.python.org/2/library/sys.html#sys.exc_info
        exc_type, exc_value, exc_traceback = sys.exc_info() # most recent (if any) by default
        
        '''
        Reason this _can_ be bad: If an (unhandled) exception happens AFTER this,
        or if we do not delete the labels on (not much) older versions of Py, the
        reference we created can linger.

        traceback.format_exc/print_exc do this very thing, BUT note this creates a
        temp scope within the function.
        '''

        traceback_details = {
                            'filename': exc_traceback.tb_frame.f_code.co_filename,
                            'lineno'  : exc_traceback.tb_lineno,
                            'name'    : exc_traceback.tb_frame.f_code.co_name,
                            'type'    : exc_type.__name__,
                            'message' : traceback.extract_tb(exc_traceback)
                            }
        
        del(exc_type, exc_value, exc_traceback) # So we don't leave our local labels/objects dangling
        # This still isn't "completely safe", though!
        # "Best (recommended) practice: replace all exc_type, exc_value, exc_traceback
        # with sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]
        
        print
        print(traceback.format_exc())
        print
        print(traceback_template % traceback_details)
        print

        #traceback.print_exception()
        raise

def prepareFeatureWithData(dataName,dataFrequency,newFeatureDf,basicDf,featureIndexStamp,variation_degree=-1,requiredMinimumCorrelation=-1):
    import numpy as np
    import pandas as pd

    from utilities.fileFolderManipulations import getJupyterRootDirectory
    from utilities.fileFolderManipulations import getParentFolder
    from utilities.fileFolderManipulations import createFolder
    from utilities.fileFolderManipulations import deleteFile
    
    
    from config.environment import setAppConfigData
    from config.environment import getAppConfigData
    
    correlation = None
    reasonableCorelation = None
    newTrainingSetDf = None
    trainableFeaturesDf = None

    try:
        
        configData = getAppConfigData()
        
        if not (isinstance(requiredMinimumCorrelation,int) or isinstance(requiredMinimumCorrelation,float))or requiredMinimumCorrelation == -1:
            requiredMinimumCorrelation = configData['requiredMinimumFeatureCorrelationWithOutputData']
        print ('requiredMinimumCorrelation is >>> '+ str(requiredMinimumCorrelation))

        if not type(variation_degree) == 'int' or variation_degree == -1:
            variation_degree = configData['variationDegreeForFeatureGeneration']

        trainableFeaturesDf = None
        featureOfInterest = newFeatureDf.columns
        if variation_degree==0:
            newTrainingSetDf=pd.concat([basicDf,newFeatureDf],axis=1)
        else:
            newTrainingSetDf = _createNewTrainingSetWithFeatureVariations(basicDf,newFeatureDf,featureOfInterest,variation_degree) 

        # return newTrainingSetDf
        correlation = newTrainingSetDf.corr()
        correlation = correlation.drop_duplicates(keep='first')

        reasonableCorelation = correlation.loc[ (np.abs(correlation['open'])>requiredMinimumCorrelation) & 
        (np.abs(correlation['high'])>requiredMinimumCorrelation) &
        (np.abs(correlation['low'])>requiredMinimumCorrelation) & 
        (np.abs(correlation['close'])>requiredMinimumCorrelation)]

        # drop duplicate features based on its correlation with 'open' attribute - this is a experimental stuff can be tried using close, high and close as well
        reasonableCorelation=reasonableCorelation.drop_duplicates(subset='open', keep='first')

        # create necessary file folder structure for storing and filtering features
        preprocessedFolderPath = '/data/' + dataName + '/processed/'+ dataFrequency + '/preProcessedData'
        preProcessedDataFilePath = preprocessedFolderPath + '/processedRawData.csv'
        #getParentFolder(preProcessedDataFilePath)
        outputFolderPath = getParentFolder(preprocessedFolderPath)

        print('preprocessedFolderPath interim test >>> ' + preprocessedFolderPath)
        print('outputFolderPath interim test >>> ' + outputFolderPath)

        featuresFolder = outputFolderPath+"/features"
        createFolder(featuresFolder)
        print('featuresFolder interim test >>> ' + featuresFolder)

        rawFeaturesFolder = featuresFolder+"/rawFeatures"
        createFolder(rawFeaturesFolder)
        print('rawFeaturesFolder interim test >>> ' + rawFeaturesFolder)

        filteredFeaturesFolder = featuresFolder+"/filteredFeatures"
        createFolder(filteredFeaturesFolder)
        print('filteredFeaturesFolder interim test >>> ' + filteredFeaturesFolder)

        correlationsFolder = featuresFolder+"/correlations"
        createFolder(correlationsFolder)
        print('correlationsFolder interim test >>> ' + correlationsFolder)

        reasonableCorrelationsFolder = correlationsFolder+"/reasonableCorrelations"
        createFolder(reasonableCorrelationsFolder)
        print('reasonableCorrelationsFolder interim test >>> ' + reasonableCorrelationsFolder)

        trainableFeaturesListFilePath = filteredFeaturesFolder+"/"+featureIndexStamp+featureOfInterest[0]+"_trainableFeaturesList.csv"
        currentFeatureListFilePath = rawFeaturesFolder+"/"+featureIndexStamp+featureOfInterest[0]+"_variations_list.csv"
        currentFeatureCorrelationListFilePath = correlationsFolder+"/"+featureIndexStamp+featureOfInterest[0]+"_variations_correlation_list.csv"
        reasonableCorelationListFilePath = reasonableCorrelationsFolder+"/"+featureIndexStamp+featureOfInterest[0]+"_variations_reasonable_correlation_list.csv"

        print('trainableFeaturesListFilePath interim test >>> ' + trainableFeaturesListFilePath)
        print('currentFeatureListFilePath interim test >>> ' + currentFeatureListFilePath)
        print('currentFeatureCorrelationListFilePath interim test >>> ' + currentFeatureCorrelationListFilePath)
        print('reasonableCorelationListFilePath interim test >>> ' + reasonableCorelationListFilePath)

        deleteFile(trainableFeaturesListFilePath)
        deleteFile(currentFeatureListFilePath)
        deleteFile(currentFeatureCorrelationListFilePath)
        deleteFile(reasonableCorelationListFilePath)

        # store output information related to current 
        print('currentFeatureListFilePath interim test >>> '+currentFeatureListFilePath)
        newTrainingSetDf.to_csv(currentFeatureListFilePath, sep=',', index=False)
        correlation.to_csv(currentFeatureCorrelationListFilePath, sep=',', index=True)
        reasonableCorelation.to_csv(reasonableCorelationListFilePath, sep=',', index=True)

        if len(reasonableCorelation.index)>4:    
            # store trainable features in global file - to be used by other training feature creation procedures    
            newFilteredTrainableFeaturesDf = newTrainingSetDf[[filteredIndex for filteredIndex in reasonableCorelation.index] ]
            trainableFeaturesDf=newFilteredTrainableFeaturesDf.drop(columns=["open","close","high","low"])    
            
            
            if not trainableFeaturesDf is None or trainableFeaturesDf.shape[1]>0:
                trainableFeaturesDf.to_csv(trainableFeaturesListFilePath, sep=',', index=False)

            # assertions
            print("newTrainingSetDf shape>>>"+str(newTrainingSetDf.shape[0])+","+str(newTrainingSetDf.shape[1]))
            print("trainableFeaturesDf shape>>>"+str(trainableFeaturesDf.shape[0])+","+str(trainableFeaturesDf.shape[1]))
                        
            autoConfigData = getAppConfigData()
            autoConfigData[dataName][dataFrequency].update({'trainableFeaturesListFile':trainableFeaturesListFilePath})
            setAppConfigData(autoConfigData)
        else:
            trainableFeaturesDf = getTrainableFeaturesListDf(trainableFeaturesListFilePath)

        
    except:
        print("Error executing method >>> ")
        # exc_type, exc_obj, exc_tb = sys.exc_info()
        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print("Unexpected error:", sys.exc_info())
        # print(exc_type, fname, exc_tb.tb_lineno)
        
        # http://docs.python.org/2/library/sys.html#sys.exc_info
        exc_type, exc_value, exc_traceback = sys.exc_info() # most recent (if any) by default
        
        '''
        Reason this _can_ be bad: If an (unhandled) exception happens AFTER this,
        or if we do not delete the labels on (not much) older versions of Py, the
        reference we created can linger.

        traceback.format_exc/print_exc do this very thing, BUT note this creates a
        temp scope within the function.
        '''

        traceback_details = {
                            'filename': exc_traceback.tb_frame.f_code.co_filename,
                            'lineno'  : exc_traceback.tb_lineno,
                            'name'    : exc_traceback.tb_frame.f_code.co_name,
                            'type'    : exc_type.__name__,
                            'message' : traceback.extract_tb(exc_traceback)
                            }
        
        del(exc_type, exc_value, exc_traceback) # So we don't leave our local labels/objects dangling
        # This still isn't "completely safe", though!
        # "Best (recommended) practice: replace all exc_type, exc_value, exc_traceback
        # with sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]
        
        print
        print(traceback.format_exc())
        print
        print(traceback_template % traceback_details)
        print

        #traceback.print_exception()
        raise
    finally:
        return correlation, reasonableCorelation ,newTrainingSetDf ,trainableFeaturesDf

def createFinalTrainingFeatureList(dataName,dataFrequency,variation_degree):
    import glob
    import pandas as pd

    from utilities.fileFolderManipulations import getJupyterRootDirectory
    from utilities.fileFolderManipulations import getParentFolder

    from config.environment import getAppConfigData
    from config.environment import setAppConfigData

    from dataPreparation.featurePreparation import doBasicOperation
    
    #caluclate the deployment directory path of the current juypter node in the operating system
    jupyterNodePath = getJupyterRootDirectory()

    _basicDf,_variation_degree,_preProcessedDataFilePath,_autoConfigData,_configFilePath = doBasicOperation(dataName,
           dataFrequency,autoConfigFileRelativePath,KEY_preProcessedDataFilePath,variation_degree)

    filteredFeaturesPath=jupyterNodePath+"/data/"+dataName+"/processed/"+dataFrequency+"/features/filteredFeatures"
    outputFinalFeatureListFilePath=jupyterNodePath+"/data/"+dataName+"/processed/"+dataFrequency+"/features/finalTrainingFeatureList.csv"
    print("filteredFeaturesFolderPath >>> " + filteredFeaturesPath)

    # creating OS queryable object for python to work with to find json files in the dataFolderPath calcuated in the previous step
    csv_pattern = os.path.join(filteredFeaturesPath,'*.csv')
    print("declared csv_pattern")

    # store all the json file paths in the dataFolderPath for further processing
    file_list = glob.glob(csv_pattern)
    print("obtained file_list")

    # creating pandas dataframe references for further modification
    trainingFeatureDF = _basicDf
    print('initialized trainingFeatureDF')

    # execution assertion/ui progress update info
    print('looping through all the files to create input data')

    file_list

    #loop through all the files in the folder and create inputRawDataDF pandas datafram
    for file in file_list:     
        print ("reading input file >>> " + file + " ...")   
        data = pd.read_csv(file)
        #data=data.values[0][0]['candles']
        trainingFeatureDF = pd.concat([trainingFeatureDF,data],axis=1) #trainingFeatureDF.append(data, ignore_index = True)
        print ("File read - SUCCESS" )

    # crate the final training list file
    print("creating finalTrainingFeatureList in location >>> " + outputFinalFeatureListFilePath)
    trainingFeatureDF.to_csv(outputFinalFeatureListFilePath)

    # update auto config file
    autoConfigData = getAppConfigData(_configFilePath)
    autoConfigData[dataName][dataFrequency].update({'finalTrainingFeaturesListFile':outputFinalFeatureListFilePath})
    setAppConfigData(_configFilePath,autoConfigData)

    print ("updated config file with data >>>> finalTrainingFeaturesListFile:"+outputFinalFeatureListFilePath)
    
    return trainingFeatureDF, outputFinalFeatureListFilePath

def getQuantityBasedFeatures(dataName,dataFrequency): 
    import pandas as pd
    import numpy as np

    from utilities.fileFolderManipulations import getJupyterRootDirectory
    from config.environment import getAppConfigData

    # Variable to hold the original source folder path which is calculated from the input relative path of the source folder (relativeDataFolderPath)
    # using various python commands like os.path.abspath and os.path.join
    jupyterNodePath = getJupyterRootDirectory()

    autoConfigData = getAppConfigData()        

    preProcessedDataFilePath=autoConfigData[dataName][dataFrequency]['preProcessedDataFilePath']

    # read the raw processed data from csv file
    df = pd.read_csv(jupyterNodePath + preProcessedDataFilePath)  

    qtyMean = np.mean(df['quantity'])
    qtyMax = np.max(df['quantity'])
    normalizedQuantityDf = (df['quantity'] - qtyMean)/qtyMax

    qtyDiffDf = df['quantity'] - df['quantity'].shift(1)
    qtyDiffMean = np.mean(qtyDiffDf)
    qtyDiffMax = np.max(qtyDiffDf)
    normalizedQtyDiffDf = (qtyDiffDf - qtyDiffMean)/qtyDiffMax

    return pd.concat([normalizedQuantityDf,normalizedQtyDiffDf],axis=1)


