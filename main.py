# Controller for data processing and LSTM Control

from LSTM.control_LSTM import control_LSTM as lstmCtrl
#from LSTM.LSTM_Model import LSTM_Model
class mainCtrl():
    def __init__(self, *args):
        self.familyMember = dict() # holds a family of LSTMs
        self.keys = ("direc", 'dataTarget')
        self.famCount = 0
        self.currTargetModel = None

    def get_lstm_id(self):
        self.famCount+=1
        return "LSTM_Number__"+str(self.famCount-1)

    def loadModelsIntoFamily(self, targetModels):
        raise NotImplementedError

def newBuild(ctrl, fileInfo):
    babyLSTMController = lstmCtrl(fileInfo) #.control_LSTM()
    #babyLSTMController.initAndBuildModel()
    _id = ctrl.get_lstm_id()
    ctrl.familyMember[_id] = babyLSTMController
    lstmConfigOptions = dict() # TODO Implement. paramater format: batch_size=300, max_iterations=1000, dropout=0.8, num_layers=3, hidden_size=120, max_grad_norm=3, mu=0.005 )

    ctrl.familyMember[_id].initAndBuildModel( lstmConfigOptions)
    print("success")

def runModel(ctrl):
    if len(ctrl.familyMember) < -1:
        print("=== Main Controller says: No models exist in the family.")
    else:
        # TODO Handle existing models here
        lstmCtrl.runModel("od", "target","new data stream")

def getDataInfoFromInput():
    """ connection between either a user or other library to control
        the data that can be accessed by the LSTM
    """
    raise NotImplementedError

def getFileInfo(ctrl):
    """ Either preset or need to implement dynamic allocation
        returns an dict that contains info on where the data lives
        and the target data (for example, could be referencing just windTurbine)

        WARNING: the keys are immutable tuple initialized in the ctrl class.
        If these keys do not match the initialization in the control_LSTM class,
        build will fail.
    """
    gettingUserInput = False

    direcKey = ctrl.keys[0]
    dataTargetKey = ctrl.keys[1]
    prep = {direcKey: "prodData", dataTargetKey: "powerGeneratedWindTurbine"}

    if gettingUserInput:
        prep = getDataInfoFromInput()

    return prep

def handle(usrInput, ctrl):
    if usrInput == "help":
        print("**** TODO: Enter list of possible cmds")
    elif usrInput == "new_build":
        newBuild(ctrl, getFileInfo(ctrl))
    elif usrInput == "run_model":
        try:
            runModel(ctrl)
        except:
            print("=== Main Controller says: Failure to run model.")
    else:
        print("=== Main Controller says: requested cmd does not exist.")

def run():
    print("$ Welcome to the Main Controller.")
    print("$ Enter 'help' for a list of cmds.")
    ctrl = mainCtrl()
    while(True):
        try:
            #usrInput = input()
            usrInput = "new_build"
            print("$ Input is: " + usrInput)
            returnedObjs = handle(usrInput, ctrl)
            z = input()
            # TODO do things with the returned objects
        except KeyboardInterrupt:
            print("=== Main Controller says: Killing processes:")
            break

if __name__ == "__main__":
    run()
