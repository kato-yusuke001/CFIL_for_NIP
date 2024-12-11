import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

class Test:
    def initialize(self, directory): #おまじない
        os.chdir(path = directory)

    def terminate(self): #おまじない
        pass
    
    def execute(self, solution):
        try:
            variable_id = solution.get_variable_id("list") # "list"はNIP側で設定している変数名
            solution.set_variable(variable_id, {1:0, 2:1, 3:2, 4:3, 5:4, 6:5}) # 辞書型は1から始めることに注意

            return solution.judge_pass() 

        except Exception as  e:
            print(type(e), e)
            return solution.judge_fail()