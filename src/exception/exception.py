import sys


class CustomeException(Exception):
    def __init__(self, message,details:sys):
        self.message = message
        _, _, exe_tb = details.exc_info()
        self.line = exe_tb.tb_lineno
        self.file = exe_tb.tb_frame.f_code.co_filename
        super().__init__(self.message)
    
    def __str__(self):
        return "Error occured in python Script name: {0} at line number: {1} with message: {2}".format(self.file, self.line, self.message)
    
# if __name__ == "__main__":
#     try:
#         raise CustomeException("This is a custom exception")
#         # a=10/0
#     except Exception as e:
#         raise CustomeException(e, sys)