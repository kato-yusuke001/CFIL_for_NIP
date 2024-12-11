from enum import Enum

class VariableType(Enum):
    INVALID = -1
    NUMERIC = 0
    STRING = 1
    POINT = 2
    ARRAY = 3


### 変数(数値)
class VariableNumericObject:
    def __init__(self, solution, name):
        self.__solution = solution
        self.__name = name
        self.__id = self.__solution.variable_id(self.__name)
        self.__variable = self.__solution.get_variable_object(self.__id)
        if VariableType(self.__variable['type']) != VariableType.NUMERIC:
            raise TypeError(f'NIP数値変数ではありません({self.__name})')

    @property
    def value(self):
        return self.__solution.get_variable_numeric(self.__id)

    @value.setter
    def value(self, value):
        if isinstance(value, (int, float)):
            self.__solution.set_variable_numeric(self.__id, value)
        else:
            raise TypeError(f'数値以外の型は代入できません({self.__name})')


### 変数(文字)
class VariableStringObject:
    def __init__(self, solution, name):
        self.__solution = solution
        self.__name = name
        self.__id = self.__solution.variable_id(self.__name)
        self.__variable = self.__solution.get_variable_object(self.__id)
        if VariableType(self.__variable['type']) != VariableType.STRING:
            raise TypeError(f'NIP文字変数ではありません({self.__name})')

    @property
    def value(self):
        return self.__solution.get_variable_string(self.__id)

    @value.setter
    def value(self, value):
        if isinstance(value, str):
            self.__solution.set_variable_string(self.__id, value)
        else:
            raise TypeError(f'文字以外の型は代入できません({self.__name})')


# 変数(配列)
class VariableArrayObject:
    def __init__(self, solution, name):
        self.__solution = solution
        self.__name = name
        self.__id = self.__solution.variable_id(name)
        self.__variable = self.__solution.get_variable_object(self.__id)
        if not bool(self.__variable['is_array']):
            raise TypeError(f'NIP変数配列ではありません({self.__name})')
        self.__array = self.InnerArray(self.__solution, self.__variable)

    @property
    def array(self):
        return self.__array
    
    @array.setter
    def array(self, array):
        if isinstance(array, self.InnerArray):
            self.__array = array
        elif isinstance(array, (list, list, LibraryObject.LibraryArrayObject.InnerLibraryArray)):
            min_len = min(len(self.__array), len(array))
            for index in range(min_len):
                self.__array[index] = array[index]
        else:
            raise TypeError(f'この型は代入できません({self.__name})')

    class InnerArray():
        def __init__(self, solution, variable):
            self.__solution = solution
            self.__id = variable['id']
            self.__len = variable['elem_count']

        def nip_list(self):    
            return list(self.__solution.get_variable(self.__id).values())

        def __nip_list_element(self, index):    
            return self.__solution.get_variable_element(self.__id, index)

        def __getitem__(self, index):
            return self.__nip_list_element(index)

        def __setitem__(self, index, value):
            self.__solution.set_variable(self.__id, {index + 1 : value})

        def __str__(self):
            return str(self.nip_list())

        def __len__(self):
            return self.__len

        def __iter__(self):
            for index in range(len(self)):
                value = self.__nip_list_element(index)
                yield value


### 変数(座標)
class VariablePointObject:
    def __init__(self, solution, name):
        self.__solution = solution
        self.__name = name
        self.__id = self.__solution.variable_id(self.__name)
        self.__variable = self.__solution.get_variable_object(self.__id)
        if VariableType(self.__variable['type']) != VariableType.POINT:
            raise TypeError(f'NIP座標変数ではありません({self.__name})')
        self.__point = self.InnerPoint(self.__solution, self.__variable)

    @property
    def point(self):
        return self.__point

    @point.setter
    def point(self, point):
        if isinstance(point, self.InnerPoint):
            self.__point = point
        elif isinstance(point, LibraryObject.LibraryPointObject.InnerLibraryPoint):
            self.__solution.set_variable_point_x(self.__id, point.x)
            self.__solution.set_variable_point_y(self.__id, point.y)
        elif isinstance(point, (list, tuple)):
            if len(point) > 1:
                self.__solution.set_variable_point_x(self.__id, point[0])
                self.__solution.set_variable_point_y(self.__id, point[1])
            else:
                raise TypeError(f'データが不足しています({self.__name})')
        else:
            raise TypeError(f'この型は代入できません({self.__name})')
        

    class InnerPoint():
        def __init__(self, solution, variable):
            self.__solution = solution
            self.__id = variable['id']
    
        @property
        def x(self):
            x, _ = self.__solution.get_variable_point(self.__id)
            return x
    
        @x.setter
        def x(self, x):
            self.__solution.set_variable_point_x(self.__id, x)

        @property
        def y(self):
            _, y = self.__solution.get_variable_point(self.__id)
            return y

        @y.setter
        def y(self, y):
            self.__solution.set_variable_point_y(self.__id, y)
        
        def __str__(self):
            x, y = self.__solution.get_variable_point(self.__id)
            return str(x) + ", " + str(y)

        def __len__(self):
            return 2

        def __iter__(self):
            for index in range(len(self)):
                x, y = self.__solution.get_variable_point(self.__id)
                if index == 0: yield x
                if index == 1: yield y


### ブロブ ###
class BlobObject():
    def __init__(self, solution, name):
        self.__solution = solution
        self.__name = name
        self.__id = self.__solution.blob_id(self.__name)
        self.__blob = self.InnerBlob(self.__solution, self.__id)

    @property
    def array(self):
        return self.__blob
    
    @array.setter
    def array(self, blob):
        if isinstance(blob, self.InnerBlob):
            self.__blob = blob
        else:
            raise TypeError(f"arrayには代入できません: {self.__name}")
        
    # ブロブ(NIP配列)
    class InnerBlob():
        def __init__(self, solution, id):
            self.__solution = solution
            self.__id = id
            self.__solution.get_blob(self.__id)

        def __nip_list_element(self, index):
            return self.InnerDict(self.__solution, self.__id, index)

        def __getitem__(self, index):
            return self.__nip_list_element(index)    

        def __str__(self):
            return str(self.__solution.get_blob(self.__id))

        def __len__(self):
            return self.__solution.get_blob_record_count(self.__id)

        def __iter__(self):
            for index in range(len(self)):
                value = self.__solution.get_blob_record(self.__id, index)
                yield value

        class InnerDict:
            def __init__(self, solution, id, row_index):
                self.__id = id
                self.__row_index = row_index
                self.__solution = solution

            def get_value( self, key):
                return self.__solution.get_blob_value(self.__id, self.__row_index, key)        

            def __getitem__(self, key):
                return self.get_value(key)

            def __setitem__(self, key, value):
                return self.__solution.set_blob_value(self.__id, self.__row_index, key, value)
            
            def __str__(self):
                return str(self.__solution.get_blob_record(self.__id, self.__row_index))
            
            def __len__(self):
                return len(self.__solution.get_blob_record(self.__id, self.__row_index))


class LibraryObject:
    def __init__(self, solution, name):
        try:
            self.__solution = solution
            self.__name = name
            self.__id = solution.library_id( name )
            value_list = solution.get_library_value_vector( self.__id )
            self.__value_dic = dict()

            # プロパティ生成
            for index in value_list:
                value_object = value_list[index]
                vidx = value_object['vidx']
                self.__value_dic[value_object['name']] = value_object
                self.add_property( vidx, value_object['name'] )

        except Exception as e:
            print(f"{self.__name} : {e}")

    def add_property(self, vidx, prop_name):
        try:
            is_array = bool(self.__value_dic[prop_name]['is_array'])
            is_point = bool(self.__value_dic[prop_name]['is_point'])
            writable = bool(self.__value_dic[prop_name]['writable'])
            readable = bool(self.__value_dic[prop_name]['readable'])

            if is_array:
                setattr(self, prop_name, self.LibraryArrayObject(self.__solution, self.__id, prop_name, writable, readable, vidx))
            elif is_point:
                setattr(self, prop_name, self.LibraryPointObject(self.__solution, self.__id, prop_name, writable, readable, vidx))
            else:
                setattr(self, prop_name, self.LibraryValueObject(self.__solution, self.__id, prop_name, writable, readable, vidx))

        except Exception as e:
            print(f"{self.__name} : {e}")


    # ライブラリ変数(配列)
    class LibraryArrayObject:
        def __init__(self, solution, library_id, prop_name, writable, readable, vidx):
            self.__prop_name = prop_name
            self.__readable = readable
            self.__writable = writable
            self.__array = self.InnerLibraryArray(solution, library_id, prop_name, writable, readable, vidx)

        @property
        def array(self):
            if not self.__readable:
                raise TypeError(f"この値は読み込めません({self.__prop_name})")
            return self.__array
        
        @array.setter
        def array(self, array):
            if not self.__writable:
                raise TypeError(f"この値は変更できません({self.__prop_name})")
            elif isinstance(array, self.InnerLibraryArray):
                self.__array = array
            if isinstance(array, (list, tuple, VariableArrayObject.InnerArray)):
                for index in range(len(array)):
                    self.__array[index] = array[index]
            else:
                raise TypeError(f'この型は代入できません({self.__prop_name})')


        # ライブラリ変数(NIP配列)
        class InnerLibraryArray():
            def __init__(self, solution, library_id, prop_name, writable, readable, vidx):
                self.__solution = solution
                self.__id = library_id
                self.__vidx = vidx
                self.__id = library_id

            def nip_list(self):    
                return list(self.__solution.get_library_value(self.__id, self.__vidx, -1).values())
            
            def __nip_list_element(self, index):    
                return self.__solution.get_library_value(self.__id, self.__vidx, index)
            
            def __getitem__(self, index):
                return self.__nip_list_element(index)

            def __setitem__(self, index, value):
                self.__solution.set_library_value(self.__id, self.__vidx, index, value)     

            def __str__(self):
                return str(self.nip_list())

            def __len__(self):
                return len(self.nip_list())

            def __iter__(self):
                for index in range(len(self)):
                    value = self.__nip_list_element(index)
                    yield value


    ### ライブラリ変数(座標)
    class LibraryPointObject:
        def __init__(self, solution, library_id, prop_name, writable, readable, vidx):
            try:
                self.__solution = solution
                self.__id = library_id
                self.__vidx = vidx
                self.__prop_name = prop_name
                self.__readable = readable
                self.__writable = writable
                self.__point = self.InnerLibraryPoint(solution, library_id, prop_name, writable, readable, vidx)
            except Exception as e:
                print(f"{prop_name} : {e}")

        @property
        def point(self):
            if not self.__readable:
                raise TypeError(f"この値は読み込めません({self.__prop_name})")
            return self.__point

        @point.setter
        def point(self, point):
            if not self.__writable:
                raise TypeError(f"この値は変更できません({self.__prop_name})")

            if isinstance(point, self.InnerLibraryPoint):
                self.__point = point
            elif isinstance(point, VariablePointObject.InnerPoint):
                self.__solution.set_library_value_point(self.__id, self.__vidx, point.x, point.y)
            elif isinstance(point, (list, tuple)):
                if len(point) > 1:
                    self.__solution.set_library_value_point(self.__id, self.__vidx, point[0], point[1])
                else:
                    raise TypeError(f'セットデータが不足しています({self.__name})')
            else:
                raise TypeError(f'この型は代入できません({self.__name})')
            

        class InnerLibraryPoint():
            def __init__(self, solution, library_id, prop_name, writable, readable, vidx):
                self.__solution = solution
                self.__id = library_id
                self.__vidx = vidx
                self.__prop_name = prop_name
        
            @property
            def x(self):
                x, _ = self.__solution.get_library_value_point(self.__id, self.__vidx)
                return x
        
            @x.setter
            def x(self, x):
                TypeError(f"この変数からX座標に対し設定することはできません({self.__prop_name})")

            @property
            def y(self):
                _, y = self.__solution.get_library_value_point(self.__id, self.__vidx)
                return y

            @y.setter
            def y(self, y):
                TypeError(f"この変数からY座標に対し設定することはできません({self.__prop_name})")
            
            def __str__(self):
                x, y = self.__solution.get_library_value_point(self.__id, self.__vidx)
                return str(x) + ", " + str(y)
      
            def __len__(self):
                return len(2)
      
            def __iter__(self):
                for index in range(len(self)):
                    x, y = self.__solution.get_library_value_point(self.__id)
                    if index == 0: yield x
                    if index == 1: yield y


    ### ライブラリ値変数(数値・文字列)
    class LibraryValueObject:
        def __init__(self, solution, library_id, prop_name, writable, readable, vidx):
            try:
                self.__solution = solution
                self.__id = library_id
                self.__prop_name = prop_name
                self.__vidx = vidx
                self.__readable = readable
                self.__writable = writable
            except Exception as e:
                print(f"{prop_name} : {e}")

        @property
        def value(self):
            if not self.__readable:
                raise RuntimeError(f'この値は読み込めません({self.__prop_name})')
            if self.__prop_name == self.__solution.get_library_total_judgement_name():
                # 総合判定のコード変換
                nip_total_judgement = self.__solution.get_library_value(self.__id, self.__vidx, -1)
                if nip_total_judgement == 1:
                    return self.__solution.JUDGE_OK
                elif nip_total_judgement == 2:
                    return self.__solution.JUDGE_NG
                else:
                    return self.__solution.JUDGE_NONE
            return self.__solution.get_library_value(self.__id, self.__vidx, -1)

        @value.setter
        def value(self, value):
            if not self.__writable:
                raise RuntimeError(f'この値は変更できません({self.__prop_name})')
            self.__solution.set_library_value(self.__id, self.__vidx, -1, value)
