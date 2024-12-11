import pynip_op_solution

class NipOpSolution:

  JUDGE_NG = 0
  JUDGE_OK = 1
  JUDGE_NONE = -1

  MODE_EDITOR = 0
  MODE_EXECUTOR = 1

  def initialize(self, instance):
    self.instance = instance
    self.__execution_data = None
    self.__total_judgement_name = pynip_op_solution.set_library_total_judgement_name()

  def set_library_total_judgement_name(self):
    return pynip_op_solution.get_library_total_judgement_name()

  def set_execution_data(self, execution_data):
    self.__execution_data = execution_data

  def get_library_total_judgement_name(self):
    return self.__total_judgement_name

  def judge_pass(self):
    return pynip_op_solution.judge_pass()

  def judge_fail(self):
    return pynip_op_solution.judge_fail()

  def error_code(self):
    return pynip_op_solution.get_error_code(self.instance)

  def solution_file_name(self):
    return pynip_op_solution.get_solution_file_name(self.instance)

  def app_directory(self):
    return pynip_op_solution.get_app_directory()

  def root_directory(self):
    return pynip_op_solution.get_root_directory()

  def mode(self):
    return pynip_op_solution.get_app_mode()

  def simulation_image_path(self):
    return pynip_op_solution.get_simulation_image_path(self.__execution_data)

  def simulation_file_name(self):
    return pynip_op_solution.get_simulation_file_name(self.__execution_data)

  def simulation_base_name(self):
    return pynip_op_solution.get_simulation_base_name(self.__execution_data)

  def image_id(self, name):
    return pynip_op_solution.get_image_id(self.instance, name)

  def image_height(self, image_id):
    return pynip_op_solution.get_image_height(self.instance, image_id)

  def image_width(self, image_id):
    return pynip_op_solution.get_image_width(self.instance, image_id)

  def roi_id(self, name):
    return pynip_op_solution.get_roi_id(self.instance, name)

  def roi_vertex(self, area_id, vertex_index):
    return pynip_op_solution.get_roi_vertex(self.instance, area_id, vertex_index)

  def roi_area(self, area_id):
    return pynip_op_solution.get_area(self.instance, area_id)

  def library_id(self, name):
    return pynip_op_solution.get_library_id(self.instance, name)

  def get_library_value_point(self, library_id, vidx):
    return pynip_op_solution.get_library_value_point(self.instance, library_id, vidx)

  def set_library_value_point(self, library_id, vidx, x, y):
    return pynip_op_solution.set_library_value_point(self.instance, library_id, vidx, x, y)

  def get_library_value(self, library_id, vidx, ele_index):
    return pynip_op_solution.get_library_value(self.instance, library_id, vidx, ele_index)

  def set_library_value(self, library_id, vidx, ele_index, value):
    return pynip_op_solution.set_library_value(self.instance, library_id, vidx, ele_index, value)

  def get_library_value_vector(self, library_id):
    return pynip_op_solution.get_library_value_vector(self.instance, library_id)

  def variable_id(self, name):
    return pynip_op_solution.get_variable_id(self.instance, name)

  def get_variable_object(self, variable_id):
    return pynip_op_solution.get_variable_object(self.instance, variable_id)

  def set_variable(self, variable_id, dict):
    return pynip_op_solution.set_variable(self.instance, variable_id, dict)

  def set_variable_numeric(self, variable_id, value):
    return pynip_op_solution.set_variable_numeric(self.instance, variable_id, value)

  def set_variable_string(self, variable_id, value):
    return pynip_op_solution.set_variable_string(self.instance, variable_id, value)

  def set_variable_point(self, variable_id, x, y):
    return pynip_op_solution.set_variable_point(self.instance, variable_id, x, y)

  def set_variable_point_x(self, variable_id, x):
    return pynip_op_solution.set_variable_point_x(self.instance, variable_id, x)

  def set_variable_point_y(self, variable_id, y):
    return pynip_op_solution.set_variable_point_y(self.instance, variable_id, y)

  def get_variable(self, variable_id):
    return pynip_op_solution.get_variable(self.instance, variable_id)

  def get_variable_numeric(self, variable_id):
    return pynip_op_solution.get_variable_numeric(self.instance, variable_id)

  def get_variable_string(self, variable_id):
    return pynip_op_solution.get_variable_string(self.instance, variable_id)

  def get_variable_point(self, variable_id):
    return pynip_op_solution.get_variable_point(self.instance, variable_id)

  def get_variable_element(self, variable_id, element_index):
    return pynip_op_solution.get_variable_element(self.instance, variable_id, element_index)

  def blob_id(self, name):
    return pynip_op_solution.get_blob_id(self.instance, name)

  def get_blob(self, blob_id):
    return pynip_op_solution.get_blob(self.instance, blob_id)

  def get_blob_value(self, blob_id, row_index, colmn_name):
    return pynip_op_solution.get_blob_value(self.instance, blob_id, row_index, colmn_name)

  def set_blob_value(self, blob_id, row_index, colmn_name, value):
    return pynip_op_solution.set_blob_value(self.instance, blob_id, row_index, colmn_name, value)

  def get_blob_record(self, blob_id, record_index):
    return pynip_op_solution.get_blob_record(self.instance, blob_id, record_index)

  def get_blob_record_count(self, blob_id):
    return pynip_op_solution.get_blob_record_count(self.instance, blob_id)
