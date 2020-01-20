from utilities import *

traceback_template = '''Traceback (most recent call last):
  File "%(filename)s", line %(lineno)s, in %(name)s
%(type)s: %(message)s\n''' # Skipping the "actual line" item

def getAppConfigData():
    import json    
    from utilities.fileFolderManipulations import getJupyterRootDirectory

    data=None
    try:
        projectRootDirectory = getJupyterRootDirectory()
        configFilePath = projectRootDirectory + "/src/config/config.json"
        print(' retrieving values configured in >>> ' + configFilePath)
        with open(configFilePath) as json_data_file:            
            data = json.load(json_data_file)
        
    except :        
        print(' error retrieving values configured in >>> ' + configFilePath)
        print(' creating new configuration file >>> ' + configFilePath)
        data = {}
        f = open(configFilePath, 'a+')  # open file in append mode
        f.write('{}')
        f.close()  
       
    finally:
        return data

def setAppConfigData(data):
    
    import json  
    import sys,traceback
    
    from utilities.fileFolderManipulations import getJupyterRootDirectory

    returnValue = False
    data_string = ''
    try:
        
        projectRootDirectory = getJupyterRootDirectory()
        configFilePath = projectRootDirectory + "/src/config/config.json"

        print(' updating config file >>> ' + configFilePath)
        data_string = json.dumps(data)
        with open(configFilePath,'a+') as json_data_file:            
            json_data_file.seek(0)
            json_data_file.write('')
            json_data_file.truncate()
            json_data_file.write(data_string)
        print(' successfully updated config file >>> (try block) ' + configFilePath + ' with data >>>' + data_string)
        returnValue = True
    except FileNotFoundError:
        
        print('creating and updating config file')
        f = open(configFilePath, 'a+')  # open file in append mode
        f.write(data_string)
        f.close()  
        print(' successfully created config file >>>  (except block)' + configFilePath + ' with data >>>' + data_string)
        returnValue = True
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
        return returnValue