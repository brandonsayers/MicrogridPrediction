# Controller for data processing and LSTM Control

from LSTM.control_LSTM import control_LSTM as lstmCtrl
#from LSTM.LSTM_Model import LSTM_Model
class mainCtrl():
    def __init__(self, *args):
        self.familyMember = dict() # holds a family of LSTMs
        self.famCount = 0
        self.currTargetModel = None

    def get_lstm_id(self):
        self.famCount+=1
        return "LSTM_Number__"+str(self.famCount-1)

    def loadModelsIntoFamily(self, targetModels):
        raise NotImplementedError

def newBuild(ctrl):
    babyLSTMController = lstmCtrl() #.control_LSTM()
    #babyLSTMController.initAndBuildModel()
    _id = ctrl.get_lstm_id()
    ctrl.familyMember[_id] = babyLSTMController
    lstmConfigOptions = dict() # TODO Implement
    ctrl.familyMember[_id].initAndBuildModel(lstmConfigOptions)
    print("success")

def runModel(ctrl):
    if len(ctrl.familyMember) < -1:
        print("=== Main Controller says: No models exist in the family.")
    else:
        # TODO Handle existing models here
        lstmCtrl.runModel("od", "target","new data stream")

def handle(usrInput, ctrl):
    if usrInput == "help":
        print("**** TODO: Enter list of possible cmds")
    elif usrInput == "new_build":
        newBuild(ctrl)
    elif usrInput == "run_model":
        runModel(ctrl)
    else:
        print("=== Main Controller says: requested cmd does not exist.")

if __name__ == "__main__":
    print("Welcome to the Main Controller.")
    print("Enter 'help' for a list of cmds.")
    ctrl = mainCtrl()
    while(True):
        try:
            #usrInput = input()
            usrInput = "run_model"
            returnedObjs = handle(usrInput, ctrl)
            z = input()
            # TODO do things with the returned objects
        except KeyboardInterrupt:
            break
            print("=== Main Controller says: Killing processes:")
