import pynip_solution

class NipSolution:

  def initialize(self, instance):
    self.instance = instance

  def judge_pass(self):
    return pynip_solution.judge_pass()

  def judge_fail(self):
    return pynip_solution.judge_fail()

  def get_image_id(self, name):
    return pynip_solution.get_image_id(self.instance, name)

  def set_image(self, image_id, buffer, bits, channels, width, height):
    return pynip_solution.set_image(self.instance, image_id, buffer, bits, channels, width, height)

  def get_image(self, image_id):
    return pynip_solution.get_image(self.instance, image_id)

  def get_variable_id(self, name):
    return pynip_solution.get_variable_id(self.instance, name)

  def set_variable(self, variable_id, dict):
    return pynip_solution.set_variable(self.instance, variable_id, dict)

  def set_variable_numeric(self, variable_id, value):
    return pynip_solution.set_variable_numeric(self.instance, variable_id, value)

  def set_variable_string(self, variable_id, value):
    return pynip_solution.set_variable_string(self.instance, variable_id, value)

  def set_variable_point(self, variable_id, x, y):
    return pynip_solution.set_variable_point(self.instance, variable_id, x, y)

  def get_variable(self, variable_id):
    return pynip_solution.get_variable(self.instance, variable_id)

  def get_variable_numeric(self, variable_id):
    return pynip_solution.get_variable_numeric(self.instance, variable_id)

  def get_variable_string(self, variable_id):
    return pynip_solution.get_variable_string(self.instance, variable_id)

  def get_variable_point(self, variable_id):
    return pynip_solution.get_variable_point(self.instance, variable_id)
