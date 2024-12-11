import os 

class SetList: #pythonからNIPの変数に値（配列）を代入する
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

class SetValue: #pythonからNIPの変数に数値を代入する
    def initialize(self, directory):
        os.chdir(path = directory)

    def terminate(self):
        pass

    def execute(self, solution):
        try:
            variable_id = solution.get_variable_id("value") # "value"はNIP側で設定している変数名
            
            solution.set_variable(variable_id, {0:1000}) #NIP側で配列じゃなくても辞書型で設定する必要がある

            return solution.judge_pass()

        except Exception as  e:
            print(type(e), e)
            return solution.judge_fail()
        
class GetList: #NIPの変数（配列）に格納されている値をpythonで使用する
    def initialize(self, directory):
        os.chdir(path = directory)

    def terminate(self):
        pass
    
    def execute(self, solution):
        try:
            variable_id = solution.get_variable_id("list")
            dst = list(solution.get_variable(variable_id).values())
            print(dst)
            return solution.judge_pass()

        except Exception as  e:
            print(type(e), e)
            return solution.judge_fail()

class SetValue:  #NIPの変数に格納されている値をpythonで使用する
    def initialize(self, directory):
        os.chdir(path = directory)

    def terminate(self):
        pass

    def execute(self, solution):
        try:
            variable_id = solution.get_variable_id("value")
            value = list(solution.get_variable(variable_id).values())
            print(value)

            return solution.judge_pass()

        except Exception as  e:
            print(type(e), e)
            return solution.judge_fail()

