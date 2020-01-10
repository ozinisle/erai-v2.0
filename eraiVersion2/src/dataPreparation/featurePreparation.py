import os, errno, traceback, sys

from pathlib import Path

traceback_template = '''Traceback (most recent call last):
  File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n''' # Skipping the "actual line" item

def doBasicOperation(dataName,dataFrequency,autoConfigFileRelativePath,KEY_preProcessedDataFilePath,variation_degree):
    import os,sys,traceback    
    from datetime import datetime, timedelta

    import pandas as pd  
    import numpy as np

    
    from utilities.environment import getAutoConfigData
    from utilities.environment import setAutoConfigData

    
    from utilities.fileFolderManipulations import getJupyterRootDirectory
    from utilities.fileFolderManipulations import getParentFolder
    from utilities.fileFolderManipulations import createFolder
    print ("into method doBasicOperation")

    try:

        # Variable to hold the original source folder path which is calculated from the input relative path of the source folder (relativeDataFolderPath)
        # using various python commands like os.path.abspath and os.path.join
        jupyterNodePath = None

        configFilePath = None    

        # holds data from input data file - Truth source, should be usd only for reference and no updates should happen to this variable
        inputRawProcessedDataDF = None    

        #caluclate the deployment directory path of the current juypter node in the operating system
        jupyterNodePath = getJupyterRootDirectory()
        print("jupyterNodePath >>> "+jupyterNodePath)

        configFilePath=jupyterNodePath+autoConfigFileRelativePath
        print("configFilePath >>> "+configFilePath)

        autoConfigData = getAutoConfigData(configFilePath)
        

        preProcessedDataFilePath=autoConfigData[dataName][dataFrequency][KEY_preProcessedDataFilePath]

        # read the raw processed data from csv file
        inputRawProcessedDataDF = pd.read_csv(preProcessedDataFilePath)  

        basicDf = createFundamentalFeatures(inputRawProcessedDataDF)
        
        print("before return statement of method doBasicOperation ")

        return basicDf,variation_degree,preProcessedDataFilePath,autoConfigData,configFilePath

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

def doFeatureAssessment(newFeatureDf,basicDf,variation_degree,preProcessedDataFilePath,autoConfigData,
    configFilePath,requiredMinimumCorrelation,featureIndexStamp,dataName,dataFrequency):
    import numpy as np
    import pandas as pd

    from utilities.fileFolderManipulations import getJupyterRootDirectory
    from utilities.fileFolderManipulations import getParentFolder
    from utilities.fileFolderManipulations import createFolder
    from utilities.fileFolderManipulations import deleteFile
    
    from utilities.environment import setAutoConfigData
    
    try:

        featureOfInterest = newFeatureDf.name
        if variation_degree==0:
            newTrainingSetDf=pd.concat([basicDf,newFeatureDf],axis=1)
        else:
            newTrainingSetDf = createNewTrainingSetWithFeatureVariations(basicDf,newFeatureDf,featureOfInterest,variation_degree) 

        correlation = newTrainingSetDf.corr()

        reasonableCorelation = correlation.loc[ (np.abs(correlation['open'])>requiredMinimumCorrelation) & 
        (np.abs(correlation['high'])>requiredMinimumCorrelation) &
        (np.abs(correlation['low'])>requiredMinimumCorrelation) & 
        (np.abs(correlation['close'])>requiredMinimumCorrelation)]

        # create necessary file folder structure for storing and filtering features
        preprocessedFolderPath = getParentFolder(preProcessedDataFilePath)
        outputFolderPath = getParentFolder(preprocessedFolderPath)

        featuresFolder = outputFolderPath+"\\features"
        createFolder(featuresFolder)

        rawFeaturesFolder = featuresFolder+"\\rawFeatures"
        createFolder(rawFeaturesFolder)

        filteredFeaturesFolder = featuresFolder+"\\filteredFeatures"
        createFolder(filteredFeaturesFolder)

        correlationsFolder = featuresFolder+"\\correlations"
        createFolder(correlationsFolder)

        reasonableCorrelationsFolder = correlationsFolder+"\\reasonableCorrelations"
        createFolder(reasonableCorrelationsFolder)

        trainableFeaturesListFilePath = filteredFeaturesFolder+"\\"+featureIndexStamp+featureOfInterest+"_trainableFeaturesList.csv"
        currentFeatureListFilePath = rawFeaturesFolder+"\\"+featureIndexStamp+featureOfInterest+"_variations_list.csv"
        currentFeatureCorrelationListFilePath = correlationsFolder+"\\"+featureIndexStamp+featureOfInterest+"_variations_correlation_list.csv"
        reasonableCorelationListFilePath = reasonableCorrelationsFolder+"\\"+featureIndexStamp+featureOfInterest+"_variations_reasonable_correlation_list.csv"

        deleteFile(trainableFeaturesListFilePath)
        deleteFile(currentFeatureListFilePath)
        deleteFile(currentFeatureCorrelationListFilePath)
        deleteFile(reasonableCorelationListFilePath)

        # store output information related to current 
        newTrainingSetDf.to_csv(currentFeatureListFilePath, sep=',', index=False)
        correlation.to_csv(currentFeatureCorrelationListFilePath, sep=',', index=True)
        reasonableCorelation.to_csv(reasonableCorelationListFilePath, sep=',', index=True)

        if len(reasonableCorelation.index)>4:    
            # store trainable features in global file - to be used by other training feature creation procedures    
            newFilteredTrainableFeaturesDf = newTrainingSetDf[[filteredIndex for filteredIndex in reasonableCorelation.index] ]
            trainableFeaturesDf=newFilteredTrainableFeaturesDf.drop(columns=["open","close","high","low"])    
            # trainableFeaturesDf = getTrainableFeaturesListDf(trainableFeaturesListFilePath)
            # if trainableFeaturesDf is None:
            #     trainableFeaturesDf= newFilteredTrainableFeaturesDf
            # else:        
            #     # newFilteredTrainableFeaturesDf=newFilteredTrainableFeaturesDf.drop(columns=["open","close","high","low"])    
            #     # trainableFeaturesDf = pd.concat([trainableFeaturesDf,newFilteredTrainableFeaturesDf],axis=1)
            #     for index in reasonableCorelation:
            #         try:
            #             trainableFeaturesDf[index] = newFilteredTrainableFeaturesDf[index]
            #         except KeyError:
            #             print ('key error >>>' + index)
            
            if not trainableFeaturesDf is None or trainableFeaturesDf.shape[1]>0:
                trainableFeaturesDf.to_csv(trainableFeaturesListFilePath, sep=',', index=False)

            # assertions
            print("newTrainingSetDf shape>>>"+str(newTrainingSetDf.shape[0])+","+str(newTrainingSetDf.shape[1]))
            print("trainableFeaturesDf shape>>>"+str(trainableFeaturesDf.shape[0])+","+str(trainableFeaturesDf.shape[1]))
            
            autoConfigData[dataName][dataFrequency].update({'trainableFeaturesListFile':trainableFeaturesListFilePath})
            setAutoConfigData(configFilePath,autoConfigData)
        else:
            trainableFeaturesDf = getTrainableFeaturesListDf(trainableFeaturesListFilePath)

        return correlation, reasonableCorelation ,newTrainingSetDf ,trainableFeaturesDf
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

def getFeatureVariations(row, featureName, variation_degree_count):
    import numpy as np

    try:
    
        featureVal=row[featureName]    
        
        for iterator in range(1,variation_degree_count+1):
            row[featureName+'_exp_1'] = np.exp(featureVal)
            row[featureName+'_exp_inv_1'] = np.exp(-1*featureVal)
        
            if iterator>1:
                val= np.power(featureVal,iterator)
                valInv = 0
                if not val==0:
                    valInv = 1/val
                    row[featureName+'_times_inv_'+str(iterator)] = 1/(val*iterator)
                # correlation of X::mY does not change for the value m and hence commenting out the following code
                # else:
                #     row[featureName+'_times_inv_'+str(iterator)] = 0

                row[featureName+'_pow_'+str(iterator)] = val
                row[featureName+'_pow_inv_'+str(iterator)] = valInv
                row[featureName+'_exp_'+str(iterator)] = np.exp(iterator*featureVal)
                row[featureName+'_exp_inv_'+str(iterator)] = np.exp(-iterator*featureVal)

                # correlation of X::mY does not change for the value m and hence commenting out the following code
                # row[featureName+'_times_'+str(iterator)] = val*iterator        

                if val>0:
                    row[featureName+'_log_times_'+str(iterator)] = np.log(val*iterator)
                elif val<0:
                    row[featureName+'_log_times_'+str(iterator)] = -np.log(-1*val*iterator)
            
        return row 
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

def createNewTrainingSetWithFeatureVariations(basicDf,newFeatureDf,featureOfInterest,variation_degree) :
    import pandas as pd
    import numpy as np
    from tqdm import tqdm

    try:
        # Create and register a new `tqdm` instance with `pandas`
        # (can use tqdm_gui, optional kwargs, etc.)
        tqdm.pandas()



        newTrainingSetDf = pd.concat([basicDf,newFeatureDf],axis=1)

        newTrainingSetDf = newTrainingSetDf.progress_apply(lambda row,
                        featureName, variation_degree_count:getFeatureVariations(row,featureOfInterest,variation_degree),axis=1,
                        args=[featureOfInterest,variation_degree])

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

def createNewTrainingSetWithFeatureVariations_v2(basicDf,newFeatureDf,featureOfInterest,variation_degree) :
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

def prepareFeatureWithData(newFeatureDf,basicDf,variation_degree,preProcessedDataFilePath,
    configFilePath,requiredMinimumCorrelation,featureIndexStamp,dataName,dataFrequency,useVersion2=False):
    import numpy as np
    import pandas as pd

    from utilities.fileFolderManipulations import getJupyterRootDirectory
    from utilities.fileFolderManipulations import getParentFolder
    from utilities.fileFolderManipulations import createFolder
    from utilities.fileFolderManipulations import deleteFile
    
    
    from utilities.environment import setAutoConfigData
    from utilities.environment import getAutoConfigData
    
    correlation = None
    reasonableCorelation = None
    newTrainingSetDf = None
    trainableFeaturesDf = None

    try:
        trainableFeaturesDf = None
        featureOfInterest = newFeatureDf.columns
        if variation_degree==0:
            newTrainingSetDf=pd.concat([basicDf,newFeatureDf],axis=1)
        else:
            if useVersion2 :
                newTrainingSetDf = createNewTrainingSetWithFeatureVariations_v2(basicDf,newFeatureDf,featureOfInterest,variation_degree) 
            else :
                newTrainingSetDf = createNewTrainingSetWithFeatureVariations(basicDf,newFeatureDf,featureOfInterest,variation_degree) 

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
        preprocessedFolderPath = getParentFolder(preProcessedDataFilePath)
        outputFolderPath = getParentFolder(preprocessedFolderPath)

        print('preprocessedFolderPath interim test >>> ' + preprocessedFolderPath)
        print('outputFolderPath interim test >>> ' + outputFolderPath)

        featuresFolder = outputFolderPath+"\\features"
        createFolder(featuresFolder)
        print('featuresFolder interim test >>> ' + featuresFolder)

        rawFeaturesFolder = featuresFolder+"\\rawFeatures"
        createFolder(rawFeaturesFolder)
        print('rawFeaturesFolder interim test >>> ' + rawFeaturesFolder)

        filteredFeaturesFolder = featuresFolder+"\\filteredFeatures"
        createFolder(filteredFeaturesFolder)
        print('filteredFeaturesFolder interim test >>> ' + filteredFeaturesFolder)

        correlationsFolder = featuresFolder+"\\correlations"
        createFolder(correlationsFolder)
        print('correlationsFolder interim test >>> ' + correlationsFolder)

        reasonableCorrelationsFolder = correlationsFolder+"\\reasonableCorrelations"
        createFolder(reasonableCorrelationsFolder)
        print('reasonableCorrelationsFolder interim test >>> ' + reasonableCorrelationsFolder)

        trainableFeaturesListFilePath = filteredFeaturesFolder+"\\"+featureIndexStamp+featureOfInterest[0]+"_trainableFeaturesList.csv"
        currentFeatureListFilePath = rawFeaturesFolder+"\\"+featureIndexStamp+featureOfInterest[0]+"_variations_list.csv"
        currentFeatureCorrelationListFilePath = correlationsFolder+"\\"+featureIndexStamp+featureOfInterest[0]+"_variations_correlation_list.csv"
        reasonableCorelationListFilePath = reasonableCorrelationsFolder+"\\"+featureIndexStamp+featureOfInterest[0]+"_variations_reasonable_correlation_list.csv"

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
                        
            autoConfigData = getAutoConfigData(configFilePath)
            autoConfigData[dataName][dataFrequency].update({'trainableFeaturesListFile':trainableFeaturesListFilePath})
            setAutoConfigData(configFilePath,autoConfigData)
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

def createFinalTrainingFeatureList(dataName,dataFrequency,autoConfigFileRelativePath,KEY_preProcessedDataFilePath,variation_degree):
    import glob
    import pandas as pd

    from utilities.fileFolderManipulations import getJupyterRootDirectory
    from utilities.fileFolderManipulations import getParentFolder

    from utilities.environment import getAutoConfigData
    from utilities.environment import setAutoConfigData

    from dataPreparation.featurePreparation import doBasicOperation
    
    #caluclate the deployment directory path of the current juypter node in the operating system
    jupyterNodePath = getJupyterRootDirectory()

    _basicDf,_variation_degree,_preProcessedDataFilePath,_autoConfigData,_configFilePath = doBasicOperation(dataName,
           dataFrequency,autoConfigFileRelativePath,KEY_preProcessedDataFilePath,variation_degree)

    filteredFeaturesPath=jupyterNodePath+"\\data\\"+dataName+"\\processed\\"+dataFrequency+"\\features\\filteredFeatures"
    outputFinalFeatureListFilePath=jupyterNodePath+"\\data\\"+dataName+"\\processed\\"+dataFrequency+"\\features\\finalTrainingFeatureList.csv"
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
    autoConfigData = getAutoConfigData(_configFilePath)
    autoConfigData[dataName][dataFrequency].update({'finalTrainingFeaturesListFile':outputFinalFeatureListFilePath})
    setAutoConfigData(_configFilePath,autoConfigData)

    print ("updated config file with data >>>> finalTrainingFeaturesListFile:"+outputFinalFeatureListFilePath)
    
    return trainingFeatureDF, outputFinalFeatureListFilePath

