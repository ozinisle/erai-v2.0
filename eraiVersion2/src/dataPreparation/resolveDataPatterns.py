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
    from config.environment import getAppConfigData
    from config.environment import setAppConfigData

    redCandlesBySizeDf = None
    greenCandlesBySizeDf = None
    redCandlesBySizeTimesMagnitudeDf = None
    greenCandlesBySizeTimesMagnitudeDf = None
    redGreenCandlesTanhDf = None
    redGreenCandlesTanhTimesMagnitudeDf = None

    try:
    
        if boundaryValues is None :
            print('boundary values is none')
            #caluclate the deployment directory path of the current juypter node in the operating system
            jupyterNodePath = getJupyterRootDirectory()
            print("jupyterNodePath >>> "+jupyterNodePath)

            configFilePath=jupyterNodePath+autoConfigFileRelativePath
            print("configFilePath >>> "+configFilePath)

            autoConfigData = getAppConfigData()

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
                setAppConfigData(autoConfigData)
            else: 
                print('using configured boundary values - do not update configurations unless u r absolutely sure of it')
        else:
            print('using boundary values provided as parameter')

        dataMagnitudeDf = np.divide(np.sqrt(np.sum(np.square(df[['open','close','high','low']]),axis=1)),4)

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

        redGreenCandlesTanhDf = (df['close']-df['open']).rename('redGreenCandlesTanh')
        redGreenCandlesTanhDf[redGreenCandlesTanhDf<0] = -1
        redGreenCandlesTanhDf[redGreenCandlesTanhDf>0] = 1
        redGreenCandlesTanhTimesMagnitudeDf = np.multiply(redGreenCandlesTanhDf,dataMagnitudeDf)
        
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
                     greenCandlesBySizeTimesMagnitudeDf,
                     redGreenCandlesTanhDf,
                     redGreenCandlesTanhTimesMagnitudeDf],axis=1)

def getSimpleDataPatternBasedFeatures(df,relativeOpenCloseValuePercent_config=1,shortLegBoundary_config=1):
    import pandas as pd
    import numpy as np
    
    calcDf = pd.concat([
        df
    ],axis=1)
    
    calcDf['legLength'] = 0
    calcDf['legLength'] = calcDf['high']-calcDf['low']  

    calcDf['halfLegLength'] = 0
    calcDf['halfLegLength'] = np.multiply(calcDf['legLength'],0.5)

    calcDf['oneThirdLegLength'] = 0
    calcDf['oneThirdLegLength'] = np.multiply(calcDf['legLength'],1/3)

    calcDf['legLength30Percent'] = 0
    calcDf['legLength30Percent'] = np.multiply(calcDf['legLength'],0.3)

    calcDf['midLeg'] = 0
    calcDf['midLeg'] = np.divide(np.add(calcDf['high'],calcDf['low']),2)

    calcDf['deviation'] = 0
    calcDf['deviation'] =  np.multiply(calcDf['legLength'],.05)

    calcDf['twoTimesDeviation'] = 0
    calcDf['twoTimesDeviation'] =  np.multiply(calcDf['legLength'],.1)

    calcDf['is_green'] = 0
    calcDf['is_green'][calcDf['close']- calcDf['open']>0] =  1

    calcDf['is_red'] = 0
    calcDf['is_red'][calcDf['close']- calcDf['open']<0] =  1

    calcDf['openCloseDiffAbsolute']= 0
    calcDf['openCloseDiffAbsolute']=  np.absolute(np.subtract(calcDf['open'],calcDf['close']))

    calcDf['relativeOpenClose'] = 0
    calcDf['relativeOpenClose'] = np.divide(calcDf['openCloseDiffAbsolute'],calcDf['close'])

    calcDf['relativeOpenCloseValuePercent'] = 0
    calcDf['relativeOpenCloseValuePercent'] = np.multiply(calcDf['relativeOpenClose'],100)
  
    # 1.Doji 
    # Formed when opening and closing prices are virtually the same    
    calcDf['is_doji']=0
    calcDf['is_doji'][calcDf['relativeOpenCloseValuePercent'] <= relativeOpenCloseValuePercent_config] = 1
    
    # 2.isLongLeggedCandle    
    # Long Legged Candle
    # determine if this is a relative long legged candle
    calcDf['isLongLeggedCandle']=0
    calcDf['isLongLeggedCandle'][(calcDf['high']-calcDf['low'])>= shortLegBoundary_config] = 1

    # 3.is_longLeggedDoji    
    # Long-Legged Doji 
    # Consists of a Doji with very long upper and lower shadows. Indicates strong forces balanced in opposition
    calcDf['is_longLeggedDoji'] = 0
    calcDf['start_boundary'] = 0
    calcDf['end_boundary'] = 0

    calcDf['start_boundary'] = np.subtract(calcDf['midLeg'],calcDf['deviation'])
    calcDf['end_boundary'] = np.add(calcDf['midLeg'],calcDf['deviation'])
    calcDf['is_longLeggedDoji'][
        (calcDf['is_doji'] == 1) & 
        (calcDf['isLongLeggedCandle'] == 1) & 
        (calcDf['open'] > calcDf['start_boundary']) &
        (calcDf['open'] < calcDf['end_boundary']) 
    ] = 1
    calcDf['is_longLeggedDoji'][
        (calcDf['is_doji'] == 1) & 
        (calcDf['isLongLeggedCandle'] == 1) & 
        (calcDf['close'] > calcDf['start_boundary']) &
        (calcDf['close'] < calcDf['end_boundary']) 
    ] = 1

    # 4.is_dragonFlyDoji    
    # Dragonfly Doji 
    # Formed when the opening and the closing prices are at the highest of the day. If it has a 
    # longer lower shadow it signals a more bullish trend. 
    # When appearing at market bottoms it is considered to be a reversal signal
    calcDf['is_dragonFlyDoji'] = 0
    calcDf['start_boundary'] = 0
    calcDf['end_boundary'] = 0

    calcDf['start_boundary'] = calcDf['high']
    calcDf['end_boundary'] = np.subtract(calcDf['high'],calcDf['deviation'])

    calcDf['is_dragonFlyDoji'][
        (calcDf['is_doji'] == 1) & 
        (calcDf['isLongLeggedCandle'] == 1) & 
        (calcDf['open'] <= calcDf['start_boundary']) &
        (calcDf['open'] >= calcDf['end_boundary']) 
    ] = 1
    calcDf['is_dragonFlyDoji'][
        (calcDf['is_doji'] == 1) & 
        (calcDf['isLongLeggedCandle'] == 1) & 
        (calcDf['close'] <= calcDf['start_boundary']) &
        (calcDf['close'] >= calcDf['end_boundary']) 
    ] = 1
        
    # 5.is_graveStoneDoji
    # Gravestone Doji 
    # Formed when the opening and closing prices are at the lowest of the day. 
    # If it has a longer upper shadow it signals a bearish trend. 
    # When it appears at market top it is considered a reversal signal
    calcDf['is_graveStoneDoji'] = 0
    calcDf['start_boundary'] = 0
    calcDf['end_boundary'] = 0

    calcDf['start_boundary'] = np.add(calcDf['low'],calcDf['deviation'])
    calcDf['end_boundary'] = calcDf['low']

    calcDf['is_graveStoneDoji'][
        (calcDf['is_doji'] == 1) & 
        (calcDf['isLongLeggedCandle'] == 1) & 
        (calcDf['open'] <= calcDf['start_boundary']) &
        (calcDf['open'] <= calcDf['end_boundary']) 
    ] = 1 

    calcDf['is_graveStoneDoji'][
        (calcDf['is_doji'] == 1) & 
        (calcDf['isLongLeggedCandle'] == 1) & 
        (calcDf['close'] <= calcDf['start_boundary']) &
        (calcDf['close'] <= calcDf['end_boundary']) 
    ] = 1       
    
    # 6.is_hammer
    # Hammer 
    # A black or a white candlestick that consists of a small body near the high with a little or 
    # no upper shadow and a long lower tail. Considered a bullish pattern during a downtrend
    calcDf['is_hammer'] = 0
    calcDf['start_boundary'] = 0
    calcDf['end_boundary'] = 0
    calcDf['hammer_boundary'] = 0

    calcDf['start_boundary'] = calcDf['high']
    calcDf['end_boundary'] = np.subtract(calcDf['high'],calcDf['twoTimesDeviation'])
    calcDf['hammer_boundary'] = np.subtract(calcDf['start_boundary'],calcDf['legLength30Percent'])

    calcDf['is_hammer'][
        (calcDf['is_doji'] == 0) & 
        (calcDf['is_green'] == 1) & 
        (calcDf['close'] <= calcDf['start_boundary']) &
        (calcDf['close'] >= calcDf['hammer_boundary']) &
        (calcDf['open'] >= calcDf['hammer_boundary']) 
    ] = 1 
    calcDf['is_hammer'][
        (calcDf['is_doji'] == 0) & 
        (calcDf['is_red'] == 1) & 
        (calcDf['open'] <= calcDf['start_boundary']) &
        (calcDf['open'] >= calcDf['hammer_boundary']) &
        (calcDf['close'] >= calcDf['hammer_boundary']) 
    ] = 1
                
    # 7.is_inverted_hammer
    # Inverted Hammer 
    # A black or a white candlestick in an upside-down hammer position.
    calcDf['is_inverted_hammer'] = 0
    calcDf['start_boundary'] = 0
    calcDf['end_boundary'] = 0
    calcDf['inverted_hammer_boundary'] = 0    

    calcDf['start_boundary'] = np.add(calcDf['low'],calcDf['twoTimesDeviation'])
    calcDf['end_boundary'] = calcDf['low']    
    calcDf['inverted_hammer_boundary'] = np.add(calcDf['end_boundary'],calcDf['legLength30Percent'])

    calcDf['is_inverted_hammer'][
        (calcDf['is_doji'] == 0) & 
        (calcDf['is_green'] == 1) & 
        (calcDf['open'] <= calcDf['start_boundary']) &
        (calcDf['open'] >= calcDf['end_boundary']) &
        (calcDf['close'] <= calcDf['inverted_hammer_boundary']) 
    ] = 1 
    calcDf['is_inverted_hammer'][
        (calcDf['is_doji'] == 0) & 
        (calcDf['is_red'] == 1) & 
        (calcDf['close'] <= calcDf['start_boundary']) &
        (calcDf['close'] >= calcDf['end_boundary']) &
        (calcDf['close'] <= calcDf['hammer_boundary']) 
    ] = 1
         
    # 8.Hanging Man 
    # A black or a white candlestick that consists of a small body near the high with a little or 
    # no upper shadow and a long lower tail. The lower tail should be two or three times the height of the body. 
    # Considered a bearish pattern during an uptrend.
    calcDf['is_HangingMan'] = 0
    calcDf['start_boundary'] = 0
    calcDf['end_boundary'] = 0
    calcDf['hanging_man_boundary'] = 0

    calcDf['start_boundary'] = calcDf['high'] 
    calcDf['end_boundary']  = np.subtract(calcDf['high'],calcDf['twoTimesDeviation'])
    calcDf['hanging_man_boundary'] = np.subtract(calcDf['start_boundary'],calcDf['halfLegLength'])

    calcDf['is_HangingMan'] [(calcDf['is_green']>0) & 
              (calcDf['close'] <= calcDf['start_boundary']) & 
              (calcDf['close'] >= calcDf['hanging_man_boundary']) &                        
              (calcDf['open'] >= calcDf['hanging_man_boundary'])] = 1
    
    calcDf['is_HangingMan'] [(calcDf['is_red']>0) & 
              (calcDf['open'] <= calcDf['start_boundary']) & 
              (calcDf['open'] >= calcDf['hanging_man_boundary']) &                        
              (calcDf['close'] >= calcDf['hanging_man_boundary'])] = 1    

    # 9.Marubozu 
    # A long or a normal candlestick (black or white) with no shadow or tail. 
    # The high and the lows represent the opening and the closing prices. Considered a continuation pattern.
    calcDf['is_Marubozu'] = 0
    calcDf['highLowDiffAbsolute'] = 0

    calcDf['highLowDiffAbsolute'] = np.absolute(np.subtract(calcDf['high'],calcDf['low']))
    calcDf['is_Marubozu'][calcDf['highLowDiffAbsolute']==calcDf['openCloseDiffAbsolute']] = 1               

    # 10.Shaven Head 
    # A black or a white candlestick with no upper shadow. [Compared with hammer.]
    calcDf['is_shavenHead'] = 0
    calcDf['is_shavenHead'][(calcDf['is_Marubozu']==0) & ((calcDf['open']==calcDf['high']) | (calcDf['close']==calcDf['high']))] = 1

    # 11.Shaven Bottom 
    # A black or a white candlestick with no lower tail. [Compare with Inverted Hammer.]
    calcDf['is_shavenBottom'] = 0
    calcDf['is_shavenBottom'][(calcDf['is_Marubozu']==0) & ((calcDf['open']==calcDf['low']) | (calcDf['close']==calcDf['low']))] = 1

    # 12.Long Lower Shadow 
    # A black or a white candlestick is formed with a lower tail that has a length of 2/3 or more of the total 
    # range of the candlestick. Normally considered a bullish signal when it appears around price support levels.
    calcDf['start_boundary'] = 0
    calcDf['end_boundary']  = 0
    calcDf['is_long_lower_shadow'] = 0

    calcDf['start_boundary'] = calcDf['high'] 
    calcDf['end_boundary']  = np.subtract(calcDf['high'],calcDf['oneThirdLegLength'])
    calcDf['is_long_lower_shadow'][
        (calcDf['close'] <= calcDf['start_boundary']) &
        (calcDf['close'] >= calcDf['end_boundary']) &
        (calcDf['open'] <= calcDf['start_boundary']) &
        (calcDf['open'] >= calcDf['end_boundary']) 
    ] = 1
    
    # 13.Long Upper Shadow    
    # A black or a white candlestick with an upper shadow that has a length of 2/3 or more of 
    # the total range of the candlestick. Normally considered a bearish signal when it appears around price resistance levels.    
    calcDf['start_boundary'] = 0
    calcDf['end_boundary']  = 0
    calcDf['is_long_upper_shadow'] = 0

    calcDf['end_boundary'] = calcDf['low'] 
    calcDf['start_boundary']  = np.add(calcDf['low'],calcDf['oneThirdLegLength'])

    calcDf['is_long_upper_shadow'][
        (calcDf['close'] <= calcDf['start_boundary']) &
        (calcDf['close'] >= calcDf['end_boundary']) &
        (calcDf['open'] <= calcDf['start_boundary']) &
        (calcDf['open'] >= calcDf['end_boundary']) 
    ] = 1

    # 14.Shooting Star
    # A black or a white candlestick that has a small body, a long upper shadow and a little or no lower tail. 
    # Considered a bearish pattern in an uptrend.
    calcDf['is_shooting_star'] = 0
    calcDf['is_shooting_star'][(calcDf['is_long_lower_shadow']==1) & (calcDf['is_green']==1)] = 1
    
    

    # 15. Spinning Top 
    # A black or a white candlestick with a small body. The size of shadows can vary. 
    # Interpreted as a neutral pattern but gains importance when it is part of other formations.
    calcDf['is_spinningTop'] = 0
    calcDf['highOpenDiff'] = 0
    calcDf['closeLowDiff'] = 0
    calcDf['highCloseDiff'] = 0
    calcDf['openLowDiff'] = 0
    calcDf['legLength20Percent'] = 0

    calcDf['highOpenDiff'] = np.subtract(calcDf['high'],calcDf['open'])
    calcDf['closeLowDiff'] = np.subtract(calcDf['close'],calcDf['low'])
    calcDf['highCloseDiff'] = np.subtract(calcDf['high'],calcDf['close'])
    calcDf['openLowDiff'] = np.subtract(calcDf['open'],calcDf['low'])
    calcDf['legLength20Percent']=np.multiply(calcDf['legLength'],0.2)

    calcDf['is_spinningTop'][
        (calcDf['is_Marubozu'] == 0 ) &
        (calcDf['is_doji'] == 0 ) &
        (calcDf['is_shavenHead'] == 0 ) &
        (calcDf['is_shavenBottom'] == 0 ) &
        (calcDf['is_green'] == 1 ) &
        (calcDf['highOpenDiff'] >= calcDf['legLength20Percent']) &
        (calcDf['closeLowDiff'] >= calcDf['legLength20Percent']) 
    ] = 1

    calcDf['is_spinningTop'][
        (calcDf['is_Marubozu'] == 0 ) &
        (calcDf['is_doji'] == 0 ) &
        (calcDf['is_shavenHead'] == 0 ) &
        (calcDf['is_shavenBottom'] == 0 ) &
        (calcDf['is_red'] == 1 ) &
        (calcDf['highCloseDiff'] >= calcDf['legLength20Percent']) &
        (calcDf['openLowDiff'] >= calcDf['legLength20Percent']) 
    ] = 1

    filteredFeaturesDf = pd.concat([
        # calcDf['open'],calcDf['close'],calcDf['high'],calcDf['low'],
        calcDf['is_doji'],
        calcDf['is_longLeggedDoji'],
        calcDf['is_dragonFlyDoji'],
        calcDf['is_graveStoneDoji'],
        calcDf['is_hammer'],
        calcDf['is_inverted_hammer'],
        calcDf['is_HangingMan'],
        calcDf['is_Marubozu'],
        calcDf['is_shavenHead'],
        calcDf['is_shavenBottom'],
        calcDf['is_long_lower_shadow'],
        calcDf['is_long_upper_shadow'],
        calcDf['is_shooting_star'],
        calcDf['is_spinningTop']
    ],axis=1) 

    return filteredFeaturesDf 

def getComplexDataPatternFeatures(df):
    
    import pandas as pd
    import numpy as np
    
    calcDf = pd.concat([
        df
    ],axis=1)
    
    calcDf['is_red'] = 0
    calcDf['is_red'][calcDf['open']>calcDf['close']]=1
    
    calcDf['is_green'] = 0
    calcDf['is_green'][calcDf['open']<calcDf['close']]=1
    
    calcDf['is_doji'] = 0
    calcDf['is_doji'][calcDf['open']==calcDf['close']] = 1
    
    calcDf['prevHigh'] = calcDf['high'].shift(1)
    calcDf['prevClose'] = calcDf['close'].shift(1)
    calcDf['prevOpen'] = calcDf['open'].shift(1)
    calcDf['prevLow'] = calcDf['low'].shift(1)
    
    calcDf['prev2High'] = calcDf['high'].shift(2)
    calcDf['prev2Close'] = calcDf['close'].shift(2)
    calcDf['prev2Open'] = calcDf['open'].shift(2)
    calcDf['prev2Low'] = calcDf['low'].shift(2)
    
    calcDf['prev3High'] = calcDf['high'].shift(3)
    calcDf['prev3Close'] = calcDf['close'].shift(3)
    calcDf['prev3Open'] = calcDf['open'].shift(3)
    calcDf['prev3Low'] = calcDf['low'].shift(3)
    
    calcDf['prev4High'] = calcDf['high'].shift(4)
    calcDf['prev4Close'] = calcDf['close'].shift(4)
    calcDf['prev4Open'] = calcDf['open'].shift(4)
    calcDf['prev4Low'] = calcDf['low'].shift(4)
    
    calcDf['is_prevGreen'] = 0
    calcDf['is_prevGreen'][calcDf['prevClose']>calcDf['prevOpen']] = 1
    
    calcDf['is_prev2Green'] = 0
    calcDf['is_prev2Green'][calcDf['prev2Close']>calcDf['prev2Open']] = 1
    
    calcDf['is_prev3Green'] = 0
    calcDf['is_prev3Green'][calcDf['prev3Close']>calcDf['prev3Open']] = 1
    
    calcDf['is_prev4Green'] = 0
    calcDf['is_prev4Green'][calcDf['prev4Close']>calcDf['prev4Open']] = 1
    
    calcDf['is_prevRed'] = 0
    calcDf['is_prevRed'][calcDf['prevClose']<calcDf['prevOpen']] = 1
    
    calcDf['is_prev2Red'] = 0
    calcDf['is_prev2Red'][calcDf['prev2Close']<calcDf['prev2Open']] = 1
    
    calcDf['is_prev3Red'] = 0
    calcDf['is_prev3Red'][calcDf['prev3Close']<calcDf['prev3Open']] = 1
    
    calcDf['is_prev4Red'] = 0
    calcDf['is_prev4Red'][calcDf['prev4Close']<calcDf['prev4Open']] = 1 
    
    calcDf['is_prevDoji'] = 0
    calcDf['is_prevDoji'][calcDf['prevClose']==calcDf['prevOpen']] = 1 
    
    
    # 1. Bearish Harami 
    # Consists of an unusually large white body followed by a small black body 
    # (contained within large white body). It is considered as a bearish pattern when preceded by an uptrend.
    calcDf['bearishHarami'] = 0
    calcDf['bearishHarami'][
        (calcDf['is_red'] == 1) &
        (calcDf['is_prevGreen'] == 1) &
        (calcDf['prevHigh'] > calcDf['high']) &
        (calcDf['prevLow'] < calcDf['low']) &
        (calcDf['prevOpen'] < calcDf['close']) &
        (calcDf['prevClose'] > calcDf['open']) 
    ] = 1
    
    # 2.Bearish Harami Cross
    # A large white body followed by a Doji. Considered as a reversal signal when it appears at the top.
    calcDf['bearishHaramiCross'] = 0
    calcDf['bearishHaramiCross'][        
        (calcDf['is_doji'] == 1) &
        (calcDf['is_prevGreen'] == 1) &
        (calcDf['prevHigh'] > calcDf['high']) &
        (calcDf['prevLow'] < calcDf['low']) &
        (calcDf['prevOpen'] < calcDf['close']) &
        (calcDf['prevClose'] > calcDf['open']) 
    ] = 1
    
    # 3.Bearish 3-Method Formation 
    # A long black body followed by three small bodies (normally white) 
    # and a long black body. The three white bodies are contained within the range of first black body. 
    # This is considered as a bearish continuation pattern. 
    calcDf['bearish3MethodFormation'] = 0
    calcDf['bearish3MethodFormation'][
        (calcDf['is_red'] == 1) &
        (calcDf['is_prevGreen'] == 1) &
        (calcDf['is_prev2Green'] == 1) &
        (calcDf['is_prev3Green'] == 1) &
        (calcDf['is_prev4Red'] == 1) &
        ((calcDf['prev4Open'] >= calcDf['prev3Close']) & (calcDf['prev4Close'] <= calcDf['prev3Open'])) &
        ((calcDf['prev4Open'] >= calcDf['prev2Close']) & (calcDf['prev4Close'] <= calcDf['prev2Open'])) &
        ((calcDf['prev4Open'] >= calcDf['prevClose']) & (calcDf['prev4Close'] <= calcDf['prevOpen'])) &
        ((calcDf['prev4Close'] > calcDf['close']) & (calcDf['prev4Open'] > calcDf['open']))
    ] = 1
    
    # 4.Bullish 3-Method Formation 
    # Consists of a long white body followed by three small bodies (normally black) 
    # and a long white body. The three black bodies are contained within the range of first white body. 
    # This is considered as a bullish continuation pattern.
    calcDf['bullish3MethodFormation'] = 0
    calcDf['bullish3MethodFormation'][
        (calcDf['is_green'] == 1) &
        (calcDf['is_prevRed'] == 1) &
        (calcDf['is_prev2Red'] == 1) &
        (calcDf['is_prev3Red'] == 1) &
        (calcDf['is_prev4Green'] == 1) &
        ((calcDf['prev4Close'] >= calcDf['prev3Open']) & (calcDf['prev4Open'] <= calcDf['prev3Close'])) &
        ((calcDf['prev4Close'] >= calcDf['prev2Open']) & (calcDf['prev4Open'] <= calcDf['prev2Close'])) &
        ((calcDf['prev4Close'] >= calcDf['prevOpen']) & (calcDf['prev4Open'] <= calcDf['prevClose'])) & 
        ((calcDf['prev4Close'] < calcDf['close']) & (calcDf['prev4Open'] < calcDf['open']))
    ] = 1
    
    # 5.Bullish Harami 
    # Consists of an unusually large black body followed by a small white body 
    # (contained within large black body). It is considered as a bullish pattern when preceded by a downtrend.
    calcDf['bullishHarami'] = 0
    calcDf['bullishHarami'][
        (calcDf['is_green'] == 1) &
        (calcDf['is_prevRed'] == 1) &
        (calcDf['prevHigh'] > calcDf['high']) &
        (calcDf['prevLow'] < calcDf['low']) &
        (calcDf['prevOpen'] > calcDf['close']) &
        (calcDf['prevClose'] < calcDf['open']) 
    ] = 1
    
    # 6.Bullish Harami Cross 
    # A large black body followed by a Doji. It is considered as a reversal signal when it appears at the bottom.
    calcDf['bullishHaramiCross'] = 0
    calcDf['bullishHaramiCross'][        
        (calcDf['is_doji'] == 1) &
        (calcDf['is_prevRed'] == 1) &
        (calcDf['prevHigh'] > calcDf['high']) &
        (calcDf['prevLow'] < calcDf['low']) &
        (calcDf['prevOpen'] > calcDf['close']) &
        (calcDf['prevClose'] < calcDf['open']) 
    ] = 1
    
    # 7. Dark Cloud Cover 
    # Consists of a long white candlestick followed by a black candlestick that opens above 
    # the high of the white candlestick and closes well into the body of the white candlestick. 
    # It is considered as a bearish reversal signal during an uptrend.
    calcDf['darkCloudOver'] = 0
    calcDf['darkCloudOver'][
        (calcDf['is_red'] == 1) &
        (calcDf['is_prevGreen'] == 1) &
        (calcDf['prevHigh'] < calcDf['open']) &
        (calcDf['prevClose'] > calcDf['close']) &
        (calcDf['prevOpen'] < calcDf['close']) 
    ] = 1
    
    # 8.Engulfing Bearish Line 
    # Consists of a small white body that is contained within the followed large 
    # black candlestick. When it appears at top it is considered as a major reversal signal.
    calcDf['engulfingBearishLine'] = 0
    calcDf['engulfingBearishLine'][
        (calcDf['is_red'] == 1) &
        (calcDf['is_prevGreen'] == 1) &
        (calcDf['prevClose'] < calcDf['open']) &
        (calcDf['prevOpen'] > calcDf['close']) 
    ] = 1
    
    # 9.Engulfing Bullish 
    # Consists of a small black body that is contained within the followed large white candlestick. 
    # When it appears at bottom it is interpreted as a major reversal signal.
    calcDf['engulfingBullish'] = 0
    calcDf['engulfingBullish'][
        (calcDf['is_green'] == 1) &
        (calcDf['is_prevRed'] == 1) &
        (calcDf['prevOpen'] < calcDf['close']) &
        (calcDf['prevClose'] > calcDf['open']) 
    ] = 1
    
    # 10.Evening Doji Star 
    # Consists of three candlesticks. First is a large white body candlestick followed by a Doji that 
    # gap above the white body. The third candlestick is a black body that closes well into the white body. 
    # When it appears at the top it is considered as a reversal signal. It signals more bearish trend than 
    # the evening star pattern because of the doji that has appeared between the two bodies.
    calcDf['eveningDojiStar'] = 0
    calcDf['eveningDojiStar'] [
        (calcDf['is_red'] == 1) &
        (calcDf['is_prevDoji'] == 1) &
        (calcDf['is_prev2Green'] == 1) &
        (calcDf['prevOpen']>calcDf['prev2High']) &
        (calcDf['prev2High'] < calcDf['open']) &
        (calcDf['prev2Close'] > calcDf['close']) &
        (calcDf['prev2Open'] < calcDf['close']) 
    ] = 1
    
    # 11. Evening Star 
    # Consists of a large white body candlestick followed by a small body candlestick (black or white) 
    # that gaps above the previous. The third is a black body candlestick that closes well within the large white body. 
    # It is considered as a reversal signal when it appears at top level.
    calcDf['eveningStar'] = 0
    calcDf['eveningStar'] [
        (calcDf['is_red'] == 1) &
        (calcDf['is_prevGreen'] == 1) &
        (calcDf['is_prev2Green'] == 1) &
        (calcDf['prevOpen']>calcDf['prev2High']) &
        (calcDf['prev2High'] < calcDf['open']) &
        (calcDf['prev2Close'] > calcDf['close']) &
        (calcDf['prev2Open'] < calcDf['close']) 
    ] = 1
    
    # 12. Falling Window 
    # A window (gap) is created when the high of the second candlestick is below the low of the preceding candlestick. 
    # It is considered that the window should be filled with a probable resistance.
    calcDf['fallingWindow'] = 0
    calcDf['fallingWindow'][
        (calcDf['is_green'] == 1) &
        (calcDf['is_prevRed'] == 1) &
        (calcDf['prevLow']>calcDf['high'])
    ] = 1
        
    
    # 13. Morning Doji Star 
    # Consists of a large black body candlestick followed by a Doji that occurred below the preceding candlestick. 
    # On the following day, a third white body candlestick is formed that closed well into the black body candlestick 
    # which appeared before the Doji. It is considered as a major reversal signal that is more bullish than 
    # the regular morning star pattern because of the existence of the Doji.
    calcDf['morningDojiStar'] = 0
    calcDf['morningDojiStar'] [
        (calcDf['is_green'] == 1) &
        (calcDf['is_prevDoji'] == 1) &
        (calcDf['is_prev2Red'] == 1) &
        (calcDf['prevOpen']<calcDf['prev2Low']) &        
        (calcDf['prev2Close'] > calcDf['open']) &
        (calcDf['prev2Close'] < calcDf['close']) &
        (calcDf['prev2Open'] > calcDf['close']) 
    ] = 1
    
    # 14. Morning Star 
    # Consists of a large black body candlestick followed by a small body (black or white) that occurred 
    # below the large black body candlestick. On the following day, a third white body candlestick is 
    # formed that closed well into the black body candlestick. It is considered as a major reversal signal 
    # when it appears at bottom.
    calcDf['morningStar'] = 0
    calcDf['morningStar'] [
        (calcDf['is_green'] == 1) &
        (calcDf['is_prevGreen'] == 1) &
        (calcDf['is_prev2Red'] == 1) &
        (calcDf['prevClose']<calcDf['prev2Low']) &        
        (calcDf['prev2Close'] > calcDf['open']) &
        (calcDf['prev2Close'] < calcDf['close']) &
        (calcDf['prev2Open'] > calcDf['close']) 
    ] = 1
    
    # 15. On Neckline 
    # In a downtrend, Consists of a black candlestick followed by a small body white candlestick with 
    # its close near the low of the preceding black candlestick. It is considered as a bearish pattern 
    # when the low of the white candlestick is penetrated.
    calcDf['onNeckline'] = 0
    calcDf['onNeckline'][
        (calcDf['is_green'] == 1) &
        (calcDf['is_prevRed'] == 1) &
        ((calcDf['prevOpen']-calcDf['prevClose']) > (calcDf['close']-calcDf['open'])) &
        (calcDf['close']>=calcDf['prevLow']) &
        (calcDf['close']<=calcDf['prevClose']) 
    ] = 1
    
    # 16. Three Black Crows 
    # Consists of three long black candlesticks with consecutively lower closes. 
    # The closing prices are near to or at their lows. When it appears at top it is considered as a top reversal signal.
    calcDf['nearLowClose'] = np.divide(np.add(calcDf['high'],4*calcDf['low']),5)
    calcDf['nearPrevLowClose'] = np.divide(np.add(calcDf['prevHigh'],4*calcDf['prevLow']),5)
    calcDf['nearPrev2LowClose'] = np.divide(np.add(calcDf['prev2High'],4*calcDf['prev2Low']),5)
    
    calcDf['threeBlackCrows'] = 0
    calcDf['threeBlackCrows'][
        (calcDf['is_red'] == 1) &
        (calcDf['is_prevRed'] == 1) &
        (calcDf['is_prev2Red'] == 1) &   
        (calcDf['prev2Close'] <= calcDf['nearPrev2LowClose']) &
        (calcDf['prevClose'] <= calcDf['nearPrevLowClose']) &
        (calcDf['close'] <= calcDf['nearLowClose']) &
        (calcDf['prev2Close'] > calcDf['prevClose'] ) &
        (calcDf['prevClose'] > calcDf['close'] ) 
    ]=1
    
    # 17. Three White Soldiers 
    # Consists of three long white candlesticks with consecutively higher closes. 
    # The closing prices are near to or at their highs. When it appears at bottom it is interpreted as a bottom reversal signal.
    calcDf['nearHighClose'] = np.divide(np.add(4*calcDf['high'],calcDf['low']),5)
    calcDf['nearPrevHighClose'] = np.divide(np.add(4*calcDf['prevHigh'],calcDf['prevLow']),5)
    calcDf['nearPrev2HighClose'] = np.divide(np.add(4*calcDf['prev2High'],calcDf['prev2Low']),5)
    
    calcDf['threeWhiteSoldiers'] = 0
    calcDf['threeWhiteSoldiers'][
        (calcDf['is_green'] == 1) &
        (calcDf['is_prevGreen'] == 1) &
        (calcDf['is_prev2Green'] == 1) &
        (calcDf['prev2Close'] >= calcDf['nearPrev2HighClose']) &
        (calcDf['prevClose'] >= calcDf['nearPrevHighClose']) &
        (calcDf['close'] >= calcDf['nearHighClose']) &
        (calcDf['prev2Close'] < calcDf['prevClose'] ) &
        (calcDf['prevClose'] < calcDf['close'] ) 
    ]=1
    
    # 18. Tweezer Bottoms 
    # Consists of two or more candlesticks with matching bottoms. 
    # The candlesticks may or may not be consecutive and the sizes or the colours can vary. 
    # It is considered as a minor reversal signal that becomes more important when the candlesticks form another pattern.
    calcDf['tweezerBottoms'] = 0
    calcDf['tweezerBottoms'][
        (calcDf['low'] == calcDf['prevLow']) &
        ((calcDf['low'] == calcDf['close']) | (calcDf['low'] == calcDf['open'] )) &
        ((calcDf['prevLow'] == calcDf['prevClose']) | (calcDf['prevLow'] == calcDf['prevOpen'] ))
    ] = 1
    
    # 19. Tweezer Tops 
    # Consists of two or more candlesticks with matching tops. 
    # The candlesticks may or may not be consecutive and the sizes or the colours can vary. 
    # It is considered as a minor reversal signal that becomes more important when the candlesticks form another pattern.
    calcDf['tweezerTops'] = 0
    calcDf['tweezerTops'][
        (calcDf['high'] == calcDf['prevHigh']) &
        ((calcDf['high'] == calcDf['close']) | (calcDf['high'] == calcDf['open'] )) &
        ((calcDf['prevHigh'] == calcDf['prevClose']) | (calcDf['prevHigh'] == calcDf['prevOpen'] ))
    ] = 1
    
    # 20. Doji Star 
    # Consists of a black or a white candlestick followed by a Doji that gap above or below these. 
    # It is considered as a reversal signal with confirmation during the next trading day.
    calcDf['dojiStar'] = 0
    
    # 21. Piercing Line 
    # Consists of a black candlestick followed by a white candlestick that opens lower than 
    # the low of preceding but closes more than halfway into black body candlestick. 
    # It is considered as reversal signal when it appears at bottom.
    calcDf['midPrevStick'] = np.divide(np.add(calcDf['prevHigh'],calcDf['prevLow']),2)
    calcDf['piercingLine'] = 0
    calcDf['piercingLine'][
        (calcDf['is_green'] == 1) &
        (calcDf['is_prevRed'] == 1) &
        (calcDf['open'] < calcDf['prevLow']) &
        (calcDf['close'] > calcDf['midPrevStick'])
    ] = 1
    
    # 22. Rising Window 
    # Rising Window A window (gap) is created when the low of the second candlestick 
    # is above the high of the preceding candlestick. It is considered that the window should 
    # provide support to the selling pressure.A window (gap) is created when the low of the 
    # second candlestick is above the high of the preceding candlestick. It is considered that 
    # the window should provide support to the selling pressure.
    calcDf['risingWindow'] = 0
    calcDf['risingWindow'] [
        (calcDf['open'] > calcDf['prevHigh'])
    ]= 1
    
    calcDf['test'] = 0
    calcDf['test'][
        (calcDf['is_green'] == 1) &
        (calcDf['is_prevGreen'] == 1) &
        (calcDf['is_prev2Green'] == 1)
    ] = 1
    
    filteredFeaturesDf = pd.concat([
        calcDf['open'],calcDf['close'],calcDf['high'],calcDf['low'],
        calcDf['test'],
        calcDf['risingWindow'],
        calcDf['piercingLine'],calcDf['tweezerBottoms'],calcDf['tweezerTops'],
        calcDf['threeWhiteSoldiers'],calcDf['threeBlackCrows'],calcDf['onNeckline'],
        calcDf['morningStar'],calcDf['morningDojiStar'],calcDf['fallingWindow'],
        calcDf['eveningStar'],calcDf['eveningDojiStar'],
        calcDf['engulfingBullish'],calcDf['engulfingBearishLine'],
        calcDf['darkCloudOver'],calcDf['bearishHarami'],calcDf['bullishHarami'],
        calcDf['bearishHaramiCross'], calcDf['bullishHaramiCross'],
        calcDf['bearish3MethodFormation'], calcDf['bullish3MethodFormation']        
    ],axis=1)
    
    return filteredFeaturesDf
    

## test data for getComplexDataPatternFeatures method
# testDf=pd.DataFrame({
#                 'open':[110,135,175,110,120,130,120,100,120,130,170,190,175,140,190,155,160,105,115,160,105,115,90,40,110,222,210,110,220,210,140,110,130,190,130,210,190,160,190,150,110,150,110,130,290,195,120,135,150,190,130,175,160,150,140],
#                 'close':[120,145,120,170,140,140,100,110,148,178,198,162,122,120,170,165,120,107,150,120,105,150,80,50,190,227,130,190,220,130,130,190,140,110,170,150,110,170,110,150,190,140,190,130,210,115,125,140,170,100,180,155,150,137,205],
#                 'high':[130,155,190,180,140,140,140,130,150,180,200,200,180,150,200,170,180,110,170,180,110,170,100,60,200,230,220,200,230,220,150,200,150,200,200,220,200,180,200,180,200,160,200,160,300,200,130,150,197,195,200,190,180,155,212],
#                 'low':[100,125,115,100,100,110,100,100,120,120,160,160,120,120,160,125,115,100,110,115,100,110,70,30,100,210,120,100,210,120,120,100,120,100,100,140,100,120,100,120,100,120,100,120,200,100,105,130,140,90,100,140,135,131,132]
# })
# newFeatureDf = getComplexDataPatternFeatures(testDf)
# newFeatureDf.head(10)
