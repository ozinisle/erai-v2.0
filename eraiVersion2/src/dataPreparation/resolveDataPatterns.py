from utilities import *

traceback_template = '''Traceback (most recent call last):
  File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n''' # Skipping the "actual line" item

def getRedGreenCandlesCatogizedBySizeDf(df, autoConfigFileRelativePath, dataName, dataFrequency, boundaryValues=None):
    # @Param :: boundaryValues 
    #   - should be  Array of 5 tuples 
    #   - each tuble must be a pair of negative and positive float compatible values only (-0.44,0.44)
    #   - absolute value of each elements in the tuple should be less than the previous corresponding previous entry
    #       example [(-3.44,2.44),(-3.32,1.37),(-1.11,1.01),(-0.53,0.76),(-0.02,0.019)]
    import os,sys,traceback
    import json

    import pandas as pd
    import numpy as np

    from utilities.fileFolderManipulations import getJupyterRootDirectory    
    from utilities.environment import getAutoConfigData
    from utilities.environment import setAutoConfigData

    redCandlesBySizeDf = None
    greenCandlesBySizeDf = None
    redCandlesBySizeTimesMagnitudeDf = None
    greenCandlesBySizeTimesMagnitudeDf = None

    try:
    
        if boundaryValues is None :
            print('boundary values is none')
            #caluclate the deployment directory path of the current juypter node in the operating system
            jupyterNodePath = getJupyterRootDirectory()
            print("jupyterNodePath >>> "+jupyterNodePath)

            configFilePath=jupyterNodePath+autoConfigFileRelativePath
            print("configFilePath >>> "+configFilePath)

            autoConfigData = getAutoConfigData(configFilePath)

            if not autoConfigData.get(dataName):
                autoConfigData[dataName]={}
                
            if not autoConfigData[dataName].get(dataFrequency):
                autoConfigData[dataName][dataFrequency]={}


            boundaryValues = autoConfigData[dataName][dataFrequency].get('redGreenCandleSizeBoundaries')
            if boundaryValues is None or (type(boundaryValues)=='str' and boundaryValues.strip()==''):
                print('boundary values is not configured')
                closeOpenDiffDf=(df['close']-df['open']).rename('close_open_diff')

                candlesByBodyLengthDf=closeOpenDiffDf.sort_values(axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last').reset_index(drop=True)
                sortedRedCandles=candlesByBodyLengthDf.loc[candlesByBodyLengthDf[0:]<0].reset_index(drop=True)

                candlesByBodyLengthDf=closeOpenDiffDf.sort_values(axis=0, ascending=False, inplace=False, kind='quicksort', na_position='last').reset_index(drop=True)
                sortedGreenCandles=candlesByBodyLengthDf.loc[candlesByBodyLengthDf[0:]>0].reset_index(drop=True)

                interval= np.arange(.2, 1, .2)
                indexArr= [((int)(sortedRedCandles.shape[0]*interval[itr]), (int)(sortedGreenCandles.shape[0]*interval[itr])) for itr in range(0,interval.size)]

                boundaryValues = [(sortedRedCandles[indexItr[0]-1],sortedGreenCandles[indexItr[1]-1]) for indexItr in indexArr]  
   
                autoConfigData[dataName][dataFrequency].update({'redGreenCandleSizeBoundaries':boundaryValues})

                print('pushing values to autoConfigFile >>> ' + configFilePath + ' with data '+ json.dumps(autoConfigData))
                setAutoConfigData(configFilePath,autoConfigData)
            else: 
                print('using configured boundary values - do not update configurations unless u r absolutely sure of it')
        else:
            print('using boundary values provided as parameter')

        redCandlesBySizeDf = (df['close']-df['open']).rename('redCandlesBySize')
        redCandlesBySizeDf[redCandlesBySizeDf>=0] = 0
        redCandlesBySizeDf[redCandlesBySizeDf < boundaryValues[0][0]] = 5
        redCandlesBySizeDf[redCandlesBySizeDf.between(boundaryValues[0][0], boundaryValues[1][0], inclusive=True)] = 4
        redCandlesBySizeDf[redCandlesBySizeDf.between(boundaryValues[1][0], boundaryValues[2][0], inclusive=False)] = 3
        redCandlesBySizeDf[redCandlesBySizeDf.between(boundaryValues[2][0], boundaryValues[3][0], inclusive=True)] = 2
        redCandlesBySizeDf[redCandlesBySizeDf.between(boundaryValues[3][0], 0, inclusive=False)] = 1


        greenCandlesBySizeDf = (df['close']-df['open']).rename('greenCandlesBySize')
        greenCandlesBySizeDf[greenCandlesBySizeDf<=0] = 0
        greenCandlesBySizeDf[greenCandlesBySizeDf > boundaryValues[0][1]] = 5
        greenCandlesBySizeDf[greenCandlesBySizeDf.between(boundaryValues[1][1], boundaryValues[0][1], inclusive=True)] = 4
        greenCandlesBySizeDf[greenCandlesBySizeDf.between(boundaryValues[2][1], boundaryValues[1][1], inclusive=False)] = 3
        greenCandlesBySizeDf[greenCandlesBySizeDf.between(boundaryValues[3][1], boundaryValues[2][1], inclusive=True)] = 2
        greenCandlesBySizeDf[greenCandlesBySizeDf.between(0, boundaryValues[3][1], inclusive=False)] = 1
        
        dataMagnitudeDf = np.divide(np.sqrt(np.sum(np.square(df[['open','close','high','low']]),axis=1)),4)
        
        redCandlesBySizeTimesMagnitudeDf = -np.multiply(redCandlesBySizeDf,dataMagnitudeDf)
        greenCandlesBySizeTimesMagnitudeDf = np.multiply(greenCandlesBySizeDf,dataMagnitudeDf)

        redCandlesBySizeTimesMagnitudeDf = redCandlesBySizeTimesMagnitudeDf.rename('redCandlesBySizeTimesMagnitude')
        greenCandlesBySizeTimesMagnitudeDf = greenCandlesBySizeTimesMagnitudeDf.rename('greenCandlesBySizeTimesMagnitude')
    
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

        return pd.concat([redCandlesBySizeDf,greenCandlesBySizeDf, 
                      redCandlesBySizeTimesMagnitudeDf,
                     greenCandlesBySizeTimesMagnitudeDf],axis=1)

def getDojiFeatures(df):
    import pandas as pd    
    import numpy as np

    dojiDf=pd.concat([
        df[['open','close','high','low']],
        (df['open']-df['close']).rename('isDoji'),
        df['open'].rename('dragonFlyDoji'),
        df['open'].rename('graveStoneDoji'),
        df['open'].rename('dojiTrend'),
                     
        df['open'].rename('scarcityByDoji'),
        df['open'].rename('scarcityByDragonFlyDoji'),
        df['open'].rename('scarcityByGraveStoneDoji'),
        df['open'].rename('scarcityByDojiTrend'),
                     
        df['open'].rename('magnitudeScarcityProductByDoji'),
        df['open'].rename('magnitudeScarcityProductByDragonFlyDoji'),
        df['open'].rename('magnitudeScarcityProductByGraveStoneDoji'),
        df['open'].rename('magnitudeScarcityProductByDojiTrend')
    ],axis=1)    
        
    dojiDf['scarcityByDoji']=0
    dojiDf['scarcityByDragonFlyDoji']=0
    dojiDf['scarcityByGraveStoneDoji']=0
    
    dojiDf['magnitudeScarcityProductByDoji']=0
    dojiDf['magnitudeScarcityProductByDragonFlyDoji']=0
    dojiDf['magnitudeScarcityProductByGraveStoneDoji']=0

    dojiDf['isDoji']=dojiDf['isDoji'] == 0
    dojiDf['isDoji'][dojiDf['isDoji'] == False]=0
    dojiDf['isDoji'][dojiDf['isDoji'] == True]=1
    
    m=dojiDf.shape[0]
    matches_isDoji=dojiDf.loc[dojiDf['isDoji']==1]
    matches_not_isDoji=dojiDf.loc[dojiDf['isDoji']==0]
    
    dojiDf['scarcityByDoji'][dojiDf['isDoji'] == 1]=100*(m-matches_isDoji.shape[0])/m    
    dojiDf['scarcityByDoji'][dojiDf['isDoji'] == 0]=100*(m-matches_not_isDoji.shape[0])/m

    matches_dragonFlyDoji = dojiDf.loc[dojiDf['dragonFlyDoji']==1]
    matches_not_dragonFlyDoji = dojiDf.loc[dojiDf['dragonFlyDoji']==0]
    
    dojiDf['scarcityByDragonFlyDoji'][dojiDf['dragonFlyDoji'] == 1]=100*(m-matches_dragonFlyDoji.shape[0])/m    
    dojiDf['scarcityByDragonFlyDoji'][dojiDf['dragonFlyDoji'] == 0]=100*(m-matches_not_dragonFlyDoji.shape[0])/m

    matches_graveStoneDoji=dojiDf.loc[dojiDf['graveStoneDoji']==1]
    matches_not_graveStoneDoji=dojiDf.loc[dojiDf['graveStoneDoji']==0]
    
    dojiDf['scarcityByGraveStoneDoji'][dojiDf['graveStoneDoji'] == 1]=100*(m-matches_graveStoneDoji.shape[0])/m    
    dojiDf['scarcityByGraveStoneDoji'][dojiDf['graveStoneDoji'] == 0]=100*(m-matches_not_graveStoneDoji.shape[0])/m
    
    dojiDf['dragonFlyDoji'] = 0
    dojiDf['dragonFlyDoji'][(dojiDf['isDoji']==1) & (dojiDf['open']==dojiDf['high'])] = 1

    dojiDf['graveStoneDoji'] = 0
    dojiDf['graveStoneDoji'][(dojiDf['isDoji']==1) & (dojiDf['open']==dojiDf['low'])] = 1
    
    dataMagnitudeDf= np.divide(np.sqrt(np.sum(np.square(dojiDf[['open','close','high','low']]),axis=1)),4)
    dojiDf['magnitudeScarcityProductByDoji']= np.multiply(dojiDf['scarcityByDoji'],dataMagnitudeDf)/100
    dojiDf['magnitudeScarcityProductByDragonFlyDoji']=np.multiply(dojiDf['scarcityByDragonFlyDoji'],dataMagnitudeDf)/100
    dojiDf['magnitudeScarcityProductByGraveStoneDoji']=np.multiply(dojiDf['scarcityByGraveStoneDoji'],dataMagnitudeDf)/100
    
    dojiDf['dojiTrend'] = 0
    dojiDf['dojiTrend'][dojiDf['isDoji']==1] = 0.1
    dojiDf['dojiTrend'][dojiDf['dragonFlyDoji']==1] = 1
    dojiDf['dojiTrend'][dojiDf['graveStoneDoji']==1] = -1
    
    matches_dojiTrend_mild=dojiDf.loc[dojiDf['dojiTrend']==0.1]
    matches_dojiTrend_postive=dojiDf.loc[dojiDf['dojiTrend']==1]
    matches_dojiTrend_negative=dojiDf.loc[dojiDf['dojiTrend']==-1]
    matches_not_dojiTrend=dojiDf.loc[dojiDf['dojiTrend']==0]
    dojiDf['scarcityByDojiTrend']=0
    dojiDf['scarcityByDojiTrend'][dojiDf['dojiTrend'] == 0.1]=100*(m-matches_dojiTrend_mild.shape[0])/m    
    dojiDf['scarcityByDojiTrend'][dojiDf['dojiTrend'] == 1]=100*(m-matches_dojiTrend_postive.shape[0])/m    
    dojiDf['scarcityByDojiTrend'][dojiDf['dojiTrend'] == -1]=100*(m-matches_dojiTrend_negative.shape[0])/m    
    dojiDf['scarcityByDojiTrend'][dojiDf['dojiTrend'] == 0]=100*(m-matches_not_dojiTrend.shape[0])/m
    
    dojiDf['magnitudeScarcityProductByDojiTrend']=0
    dojiDf['magnitudeScarcityProductByDojiTrend']=np.multiply(dojiDf['scarcityByDojiTrend'],dataMagnitudeDf)/100

    return pd.concat([dojiDf[['isDoji','dragonFlyDoji','graveStoneDoji',
                             'dojiTrend','scarcityByDojiTrend','magnitudeScarcityProductByDojiTrend',
                             'scarcityByDoji','scarcityByDragonFlyDoji','scarcityByGraveStoneDoji',
                             'magnitudeScarcityProductByDoji','magnitudeScarcityProductByDragonFlyDoji',
                             'magnitudeScarcityProductByGraveStoneDoji']]
                     ],axis=1)


